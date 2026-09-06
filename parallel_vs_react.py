import os
import re
import json
import time
import math
import random
import string
import argparse
import threading
import traceback
import unicodedata

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

DEFAULT_MODEL = "google/gemini-2.5-flash"

SEED = 42

DEFAULT_N = 100

MAX_SEARCH_RESULTS = 5

MAX_DOC_CHARS = 1800

MAX_REACT_STEPS = 6

# 병렬 decomposition에서 최대 몇 개 subquery
MAX_PARALLEL_QUERIES = 4

# 전체 QA sample 동시 실행 개수
DEFAULT_WORKERS = 20

# 한 sample 내 parallel search worker
SEARCH_WORKERS = 4

MAX_RETRIES = 6
RETRY_BASE_SECONDS = 1.5

DEBUG = False


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

_local = threading.local()


def get_session():

    if not hasattr(_local, "session"):

        session = requests.Session()

        session.headers.update({
            "User-Agent":
                "parallel-vs-react-hotpotqa/0.1"
        })

        _local.session = session

    return _local.session


# ============================================================
# STATS
# ============================================================

class RunStats:

    def __init__(self):

        self.llm_calls = 0
        self.search_calls = 0

        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.llm_latency = 0.0
        self.search_latency = 0.0

        self.executed_steps = 0

        # 실제 search들의 latency를 합친 값
        self.sum_individual_search_latency = 0.0

    def to_dict(self):

        return vars(self).copy()


# ============================================================
# OPENROUTER
# ============================================================

def call_llm(
    messages,
    stats,
    model,
    max_tokens=500,
    temperature=0.0,
):

    last_error = None

    for attempt in range(MAX_RETRIES):

        start = time.perf_counter()

        try:

            response = requests.post(
                OPENROUTER_URL,

                headers={
                    "Authorization":
                        f"Bearer {OPENROUTER_API_KEY}",

                    "Content-Type":
                        "application/json",
                },

                json={
                    "model": model,

                    "messages": messages,

                    "temperature":
                        temperature,

                    "max_tokens":
                        max_tokens,

                    # reasoning OFF
                    "reasoning": {
                        "enabled": False
                    },
                },

                timeout=120,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            stats.llm_calls += 1
            stats.llm_latency += elapsed

            # --------------------------------------------
            # success
            # --------------------------------------------

            if response.status_code == 200:

                data = response.json()

                usage = (
                    data.get(
                        "usage",
                        {}
                    )
                    or {}
                )

                stats.prompt_tokens += (
                    usage.get(
                        "prompt_tokens",
                        0
                    )
                    or 0
                )

                stats.completion_tokens += (
                    usage.get(
                        "completion_tokens",
                        0
                    )
                    or 0
                )

                content = (
                    data["choices"][0]
                    ["message"]["content"]
                )

                if DEBUG:

                    print()
                    print("[LLM]")
                    print(content[:2000])

                return content

            # --------------------------------------------
            # retry
            # --------------------------------------------

            if (
                response.status_code == 429
                or
                response.status_code >= 500
            ):

                last_error = (
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

                wait = (
                    RETRY_BASE_SECONDS
                    *
                    (2 ** attempt)
                )

                print(
                    f"[LLM RETRY] "
                    f"status="
                    f"{response.status_code} "
                    f"attempt="
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES} "
                    f"sleep={wait:.1f}"
                )

                time.sleep(wait)

                continue

            raise RuntimeError(
                f"OpenRouter "
                f"{response.status_code}: "
                f"{response.text[:2000]}"
            )

        except requests.RequestException as e:

            last_error = repr(e)

            wait = (
                RETRY_BASE_SECONDS
                *
                (2 ** attempt)
            )

            time.sleep(wait)

    raise RuntimeError(
        f"OpenRouter failed: "
        f"{last_error}"
    )


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:

        return json.loads(text)

    except Exception:

        pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")

    if (
        obj_start >= 0
        and
        obj_end > obj_start
    ):

        candidate = text[
            obj_start:
            obj_end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:

            pass

    raise ValueError(
        f"Could not parse JSON:\n"
        f"{text[:3000]}"
    )


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

WIKI_API = (
    "https://en.wikipedia.org/w/api.php"
)


def wikipedia_search(
    query,
    stats,
    top_k=MAX_SEARCH_RESULTS,
):

    total_start = (
        time.perf_counter()
    )

    stats.search_calls += 1

    session = get_session()

    try:

        # --------------------------------------------
        # search
        # --------------------------------------------

        response = session.get(
            WIKI_API,

            params={
                "action":
                    "query",

                "format":
                    "json",

                "list":
                    "search",

                "srsearch":
                    query,

                "srlimit":
                    top_k,

                "utf8":
                    1,
            },

            timeout=30,
        )

        response.raise_for_status()

        hits = (
            response.json()
            .get(
                "query",
                {}
            )
            .get(
                "search",
                []
            )
        )

        if not hits:

            return []

        titles = [
            hit["title"]
            for hit in hits
        ]

        # --------------------------------------------
        # extracts
        # --------------------------------------------

        response2 = session.get(
            WIKI_API,

            params={
                "action":
                    "query",

                "format":
                    "json",

                "prop":
                    "extracts",

                "explaintext":
                    1,

                "redirects":
                    1,

                "titles":
                    "|".join(titles),
            },

            timeout=30,
        )

        response2.raise_for_status()

        pages = (
            response2.json()
            .get(
                "query",
                {}
            )
            .get(
                "pages",
                {}
            )
        )

        title_to_extract = {}

        for page in pages.values():

            title = page.get(
                "title",
                ""
            )

            extract = page.get(
                "extract",
                ""
            )

            title_to_extract[
                title.lower()
            ] = extract

        docs = []

        for rank, hit in enumerate(
            hits,
            start=1,
        ):

            title = hit[
                "title"
            ]

            text = (
                title_to_extract.get(
                    title.lower(),
                    "",
                )
            )

            if not text:

                text = re.sub(
                    "<.*?>",
                    " ",
                    hit.get(
                        "snippet",
                        ""
                    ),
                )

            docs.append({
                "rank":
                    rank,

                "title":
                    title,

                "text":
                    text[:MAX_DOC_CHARS],
            })

        if DEBUG:

            print()
            print(
                f"[SEARCH] {query}"
            )

            for d in docs[:3]:

                print(
                    d["rank"],
                    d["title"]
                )

        return docs

    finally:

        elapsed = (
            time.perf_counter()
            -
            total_start
        )

        stats.search_latency += (
            elapsed
        )

        stats.sum_individual_search_latency += (
            elapsed
        )


def docs_to_text(
    query,
    docs,
):

    chunks = [
        f"SEARCH QUERY: {query}"
    ]

    for doc in docs:

        chunks.append(
            (
                f"[{doc['rank']}] "
                f"{doc['title']}\n"
                f"{doc['text']}"
            )
        )

    return "\n\n".join(
        chunks
    )


# ============================================================
# ANSWER
# ============================================================

FINAL_ANSWER_PROMPT = """
Answer the original question using only the retrieved evidence.

Rules:
- Give only the shortest correct answer.
- Do not explain your reasoning.
- Do not prefix with "The answer is".
- For yes/no questions, output exactly "yes" or "no".
- For comparison questions, perform the comparison using the evidence.
- If the answer is a person/entity, output only that entity.
- Do not mention uncertainty unless absolutely necessary.

Question:
{question}

Evidence:
{evidence}
"""


def generate_final_answer(
    question,
    evidence,
    stats,
    model,
):

    return call_llm(
        [
            {
                "role":
                    "system",

                "content":
                    "You are a precise "
                    "open-domain QA system."
            },

            {
                "role":
                    "user",

                "content":
                    FINAL_ANSWER_PROMPT.format(
                        question=question,
                        evidence=evidence,
                    )
            },
        ],

        stats,

        model=model,

        max_tokens=150,
    ).strip()


# ============================================================
# METHOD 1
# VANILLA DECOMPOSE -> PARALLEL SEARCH
# ============================================================

PARALLEL_DECOMP_PROMPT = """
Decompose this comparison question into independent Wikipedia search queries.

Important:
- Generate ALL required search queries now.
- The queries will be executed simultaneously.
- Therefore no query may depend on the result of another query.
- Usually 2 queries are enough for HotpotQA comparison questions.
- Use 2 to 4 concise Wikipedia search queries.
- Do NOT answer the original question.
- Do NOT create a multi-step plan.
- Do NOT use placeholders.
- Return JSON only.

Format:

{{
  "queries": [
    "...",
    "..."
  ]
}}

Question:
{question}
"""


def run_parallel_decomposition(
    question,
    stats,
    model,
):

    # --------------------------------------------
    # one-shot decomposition
    # --------------------------------------------

    raw = call_llm(
        [
            {
                "role":
                    "user",

                "content":
                    PARALLEL_DECOMP_PROMPT.format(
                        question=question
                    )
            }
        ],

        stats,

        model=model,

        max_tokens=300,
    )

    plan = extract_json(raw)

    queries = (
        plan.get(
            "queries",
            []
        )
    )

    queries = [
        str(q).strip()
        for q in queries
        if str(q).strip()
    ]

    queries = list(
        dict.fromkeys(
            queries
        )
    )

    queries = queries[
        :MAX_PARALLEL_QUERIES
    ]

    if not queries:

        queries = [
            question
        ]

    # --------------------------------------------
    # TRUE PARALLEL SEARCH
    # --------------------------------------------

    evidence_by_index = {}

    search_wall_start = (
        time.perf_counter()
    )

    with ThreadPoolExecutor(
        max_workers=min(
            SEARCH_WORKERS,
            len(queries),
        )
    ) as executor:

        futures = {}

        for idx, query in enumerate(
            queries
        ):

            future = executor.submit(
                wikipedia_search,
                query,
                stats,
            )

            futures[
                future
            ] = (
                idx,
                query,
            )

        for future in as_completed(
            futures
        ):

            idx, query = (
                futures[future]
            )

            docs = (
                future.result()
            )

            evidence_by_index[
                idx
            ] = docs_to_text(
                query,
                docs,
            )

    parallel_search_wall_latency = (
        time.perf_counter()
        -
        search_wall_start
    )

    stats.executed_steps += (
        len(queries)
    )

    evidence = (
        "\n\n"
        "===================="
        "\n\n"
    ).join(
        evidence_by_index[i]
        for i in sorted(
            evidence_by_index
        )
    )

    answer = generate_final_answer(
        question,
        evidence,
        stats,
        model,
    )

    trace = {
        "queries":
            queries,

        "parallel_search_wall_latency":
            parallel_search_wall_latency,

        "evidence":
            evidence,
    }

    return (
        answer,
        trace,
    )


# ============================================================
# METHOD 2
# REACT
# ============================================================

REACT_PROMPT = """
You are answering a question by searching Wikipedia.

Question:
{question}

Search history:
{history}

Choose exactly ONE next action.

If you need more information:

{{
  "action": "search",
  "query": "..."
}}

If the evidence is sufficient:

{{
  "action": "finish"
}}

Rules:
- Search one query per turn.
- Use previous search results when deciding the next action.
- Do not answer the question yet.
- Avoid repeating queries.
- Return JSON only.
"""


def run_react(
    question,
    stats,
    model,
):

    trajectory = []

    evidence_parts = []

    queries = []

    for step in range(
        MAX_REACT_STEPS
    ):

        if trajectory:

            history = (
                "\n\n".join(
                    trajectory
                )
            )

        else:

            history = "(none)"

        raw = call_llm(
            [
                {
                    "role":
                        "user",

                    "content":
                        REACT_PROMPT.format(
                            question=question,
                            history=history,
                        )
                }
            ],

            stats,

            model=model,

            max_tokens=250,
        )

        action = extract_json(
            raw
        )

        action_type = (
            action.get(
                "action",
                ""
            )
        )

        if (
            action_type
            ==
            "finish"
        ):

            break

        if (
            action_type
            !=
            "search"
        ):

            break

        query = str(
            action.get(
                "query",
                ""
            )
        ).strip()

        if not query:

            break

        # duplicate query stop
        if query in queries:

            break

        queries.append(
            query
        )

        docs = wikipedia_search(
            query,
            stats,
        )

        evidence = docs_to_text(
            query,
            docs,
        )

        evidence_parts.append(
            evidence
        )

        trajectory.append(
            (
                f"ACTION:\n"
                f"search({query})\n\n"
                f"OBSERVATION:\n"
                f"{evidence}"
            )
        )

        stats.executed_steps += 1

    evidence = (
        "\n\n"
        "===================="
        "\n\n"
    ).join(
        evidence_parts
    )

    answer = generate_final_answer(
        question,
        evidence,
        stats,
        model,
    )

    trace = {
        "queries":
            queries,

        "trajectory":
            trajectory,

        "evidence":
            evidence,
    }

    return (
        answer,
        trace,
    )


# ============================================================
# METRICS
# HotpotQA style
# ============================================================

def normalize_answer(s):

    def remove_articles(text):

        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def remove_punc(text):

        exclude = set(
            string.punctuation
        )

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    def white_space_fix(text):

        return " ".join(
            text.split()
        )

    s = (
        unicodedata.normalize(
            "NFD",
            str(s)
        )
        .lower()
    )

    return white_space_fix(
        remove_articles(
            remove_punc(s)
        )
    )


def exact_match(
    prediction,
    gold,
):

    return int(
        normalize_answer(
            prediction
        )
        ==
        normalize_answer(
            gold
        )
    )


def f1_score(
    prediction,
    gold,
):

    pred = (
        normalize_answer(
            prediction
        )
        .split()
    )

    gold = (
        normalize_answer(
            gold
        )
        .split()
    )

    if (
        not pred
        or
        not gold
    ):

        return float(
            pred == gold
        )

    common = (
        Counter(pred)
        &
        Counter(gold)
    )

    same = sum(
        common.values()
    )

    if same == 0:

        return 0.0

    precision = (
        same / len(pred)
    )

    recall = (
        same / len(gold)
    )

    return (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


# ============================================================
# DATASET
# ============================================================

def load_hotpot_comparison(
    n=100,
    seed=SEED,
):

    print()
    print("=" * 100)
    print(
        "LOAD HOTPOTQA "
        "COMPARISON SUBSET"
    )
    print("=" * 100)

    # validation = 7405 examples
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    print(
        "all validation:",
        len(ds)
    )

    # ONLY COMPARISON
    ds = ds.filter(
        lambda x:
            x["type"]
            ==
            "comparison"
    )

    print(
        "comparison:",
        len(ds)
    )

    # deterministic
    ds = ds.shuffle(
        seed=seed
    )

    if n > len(ds):

        n = len(ds)

    ds = ds.select(
        range(n)
    )

    samples = []

    for row in ds:

        samples.append({
            "id":
                row["id"],

            "question":
                row["question"],

            "answer":
                row["answer"],

            "type":
                row["type"],

            "level":
                row["level"],
        })

    print()
    print(
        f"selected={len(samples)} "
        f"seed={seed}"
    )

    print()
    print("[EXAMPLES]")

    for x in samples[:5]:

        print()
        print(
            "Q:",
            x["question"]
        )

        print(
            "A:",
            x["answer"]
        )

        print(
            "level:",
            x["level"]
        )

    return samples


# ============================================================
# SAVE EVAL MANIFEST
# ============================================================

def save_manifest(
    samples,
    output_dir,
):

    path = os.path.join(
        output_dir,
        "eval_manifest.csv",
    )

    pd.DataFrame(
        samples
    ).to_csv(
        path,
        index=False,
    )

    print()
    print(
        "[MANIFEST SAVED]",
        path
    )


# ============================================================
# ONE SAMPLE
# ============================================================

def run_one_sample(
    method,
    sample,
    model,
):

    stats = RunStats()

    start = (
        time.perf_counter()
    )

    try:

        if (
            method
            ==
            "parallel_decomp"
        ):

            prediction, trace = (
                run_parallel_decomposition(
                    sample["question"],
                    stats,
                    model,
                )
            )

        elif (
            method
            ==
            "react"
        ):

            prediction, trace = (
                run_react(
                    sample["question"],
                    stats,
                    model,
                )
            )

        else:

            raise ValueError(
                method
            )

        wall_latency = (
            time.perf_counter()
            -
            start
        )

        em = exact_match(
            prediction,
            sample["answer"],
        )

        f1 = f1_score(
            prediction,
            sample["answer"],
        )

        error = None

    except Exception:

        wall_latency = (
            time.perf_counter()
            -
            start
        )

        prediction = ""

        trace = {}

        em = 0

        f1 = 0

        error = (
            traceback.format_exc()
        )

        print()
        print(
            "[ERROR]",
            method,
            sample["id"]
        )

        print(error)

    return {
        "id":
            sample["id"],

        "type":
            sample["type"],

        "level":
            sample["level"],

        "method":
            method,

        "question":
            sample["question"],

        "gold":
            sample["answer"],

        "prediction":
            prediction,

        "em":
            em,

        "f1":
            f1,

        "wall_latency":
            wall_latency,

        **stats.to_dict(),

        "error":
            error,

        "trace":
            json.dumps(
                trace,
                ensure_ascii=False,
            ),
    }


# ============================================================
# METHOD BENCHMARK
# ============================================================

def run_method_parallel(
    method,
    samples,
    model,
    workers,
    output_dir,
):

    print()
    print("=" * 110)

    print(
        f"METHOD={method} "
        f"N={len(samples)} "
        f"WORKERS={workers}"
    )

    print("=" * 110)

    rows_by_index = {}

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {}

        for index, sample in enumerate(
            samples
        ):

            future = executor.submit(
                run_one_sample,
                method,
                sample,
                model,
            )

            futures[
                future
            ] = index

        running_em = 0
        running_f1 = 0
        errors = 0
        count = 0

        pbar = tqdm(
            total=len(samples),
            desc=method,
        )

        for future in as_completed(
            futures
        ):

            index = futures[
                future
            ]

            row = (
                future.result()
            )

            rows_by_index[
                index
            ] = row

            count += 1

            running_em += (
                row["em"]
            )

            running_f1 += (
                row["f1"]
            )

            if row["error"]:

                errors += 1

            pbar.set_postfix(
                EM=(
                    f"{running_em/count:.3f}"
                ),
                F1=(
                    f"{running_f1/count:.3f}"
                ),
                err=errors,
            )

            pbar.update(1)

    pbar.close()

    rows = [
        rows_by_index[i]
        for i in range(
            len(samples)
        )
    ]

    df = pd.DataFrame(
        rows
    )

    path = os.path.join(
        output_dir,
        f"{method}.csv",
    )

    df.to_csv(
        path,
        index=False,
    )

    print()
    print("[RESULT]")

    print(
        "EM             :",
        df["em"].mean()
    )

    print(
        "F1             :",
        df["f1"].mean()
    )

    print(
        "search calls   :",
        df["search_calls"].mean()
    )

    print(
        "LLM calls      :",
        df["llm_calls"].mean()
    )

    print(
        "wall latency   :",
        df["wall_latency"].mean()
    )

    print(
        "LLM latency    :",
        df["llm_latency"].mean()
    )

    print(
        "search latency :",
        df["search_latency"].mean()
    )

    print(
        "steps          :",
        df["executed_steps"].mean()
    )

    print(
        "errors         :",
        df["error"].notna().sum()
    )

    print(
        "saved          :",
        path
    )

    return df


# ============================================================
# PAIRED COMPARISON
# ============================================================

def paired_analysis(
    parallel_df,
    react_df,
    output_dir,
):

    a = (
        parallel_df[
            [
                "id",
                "em",
                "f1",
                "prediction",
                "wall_latency",
                "search_calls",
                "llm_calls",
            ]
        ]
        .rename(
            columns={
                "em":
                    "parallel_em",

                "f1":
                    "parallel_f1",

                "prediction":
                    "parallel_prediction",

                "wall_latency":
                    "parallel_latency",

                "search_calls":
                    "parallel_search_calls",

                "llm_calls":
                    "parallel_llm_calls",
            }
        )
    )

    b = (
        react_df[
            [
                "id",
                "em",
                "f1",
                "prediction",
                "wall_latency",
                "search_calls",
                "llm_calls",
            ]
        ]
        .rename(
            columns={
                "em":
                    "react_em",

                "f1":
                    "react_f1",

                "prediction":
                    "react_prediction",

                "wall_latency":
                    "react_latency",

                "search_calls":
                    "react_search_calls",

                "llm_calls":
                    "react_llm_calls",
            }
        )
    )

    merged = a.merge(
        b,
        on="id",
        how="inner",
    )

    merged[
        "parallel_only_correct"
    ] = (
        (merged["parallel_em"] == 1)
        &
        (merged["react_em"] == 0)
    )

    merged[
        "react_only_correct"
    ] = (
        (merged["parallel_em"] == 0)
        &
        (merged["react_em"] == 1)
    )

    merged[
        "both_correct"
    ] = (
        (merged["parallel_em"] == 1)
        &
        (merged["react_em"] == 1)
    )

    merged[
        "both_wrong"
    ] = (
        (merged["parallel_em"] == 0)
        &
        (merged["react_em"] == 0)
    )

    path = os.path.join(
        output_dir,
        "paired.csv",
    )

    merged.to_csv(
        path,
        index=False,
    )

    print()
    print("=" * 100)
    print("PAIRED ANALYSIS")
    print("=" * 100)

    print(
        "parallel only correct:",
        merged[
            "parallel_only_correct"
        ].sum()
    )

    print(
        "react only correct   :",
        merged[
            "react_only_correct"
        ].sum()
    )

    print(
        "both correct         :",
        merged[
            "both_correct"
        ].sum()
    )

    print(
        "both wrong           :",
        merged[
            "both_wrong"
        ].sum()
    )

    print()

    print(
        "parallel avg latency :",
        merged[
            "parallel_latency"
        ].mean()
    )

    print(
        "react avg latency    :",
        merged[
            "react_latency"
        ].mean()
    )

    print(
        "speedup              :",
        (
            merged[
                "react_latency"
            ].mean()
            /
            merged[
                "parallel_latency"
            ].mean()
        )
    )

    return merged


# ============================================================
# SUMMARY
# ============================================================

def save_summary(
    dfs,
    output_dir,
):

    rows = []

    for method, df in dfs.items():

        rows.append({
            "method":
                method,

            "n":
                len(df),

            "EM":
                df["em"].mean(),

            "F1":
                df["f1"].mean(),

            "search_calls":
                df[
                    "search_calls"
                ].mean(),

            "llm_calls":
                df[
                    "llm_calls"
                ].mean(),

            "wall_latency":
                df[
                    "wall_latency"
                ].mean(),

            "prompt_tokens":
                df[
                    "prompt_tokens"
                ].mean(),

            "completion_tokens":
                df[
                    "completion_tokens"
                ].mean(),

            "errors":
                df[
                    "error"
                ].notna().sum(),
        })

    summary = pd.DataFrame(
        rows
    )

    path = os.path.join(
        output_dir,
        "summary.csv",
    )

    summary.to_csv(
        path,
        index=False,
    )

    print()
    print("=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "parallel_decomp",
            "react",
        ],
    )

    parser.add_argument(
        "--output-dir",
        default=
            "./parallel_vs_react_results",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    global DEBUG

    DEBUG = args.debug

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print("=" * 120)
    print(
        "VANILLA PARALLEL DECOMPOSITION "
        "VS REACT"
    )
    print("=" * 120)

    print(
        "model  :",
        args.model
    )

    print(
        "n      :",
        args.n
    )

    print(
        "seed   :",
        args.seed
    )

    print(
        "workers:",
        args.workers
    )

    # ========================================================
    # FIXED EVAL SET
    # ========================================================

    samples = (
        load_hotpot_comparison(
            n=args.n,
            seed=args.seed,
        )
    )

    save_manifest(
        samples,
        args.output_dir,
    )

    dfs = {}

    # ========================================================
    # METHODS
    # ========================================================

    for method in args.methods:

        df = run_method_parallel(
            method=
                method,

            samples=
                samples,

            model=
                args.model,

            workers=
                args.workers,

            output_dir=
                args.output_dir,
        )

        dfs[
            method
        ] = df

    # ========================================================
    # SUMMARY
    # ========================================================

    save_summary(
        dfs,
        args.output_dir,
    )

    # ========================================================
    # PAIRED
    # ========================================================

    if (
        "parallel_decomp"
        in dfs
        and
        "react"
        in dfs
    ):

        paired_analysis(
            dfs[
                "parallel_decomp"
            ],

            dfs[
                "react"
            ],

            args.output_dir,
        )


if __name__ == "__main__":

    main()
