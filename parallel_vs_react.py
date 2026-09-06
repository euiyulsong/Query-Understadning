import os
import re
import json
import time
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
import numpy as np
import faiss
import torch

from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "google/gemini-2.5-flash"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
DEFAULT_N = 100

TOP_K = 5

MAX_DOC_CHARS = 1800

MAX_REACT_STEPS = 6

MAX_PARALLEL_QUERIES = 4

# sample-level parallelism
DEFAULT_WORKERS = 8

# 한 질문 내부 sub-query 병렬 retrieval
SEARCH_WORKERS = 4

DEBUG = False

OPENROUTER_MAX_RETRIES = 6
OPENROUTER_RETRY_BASE = 1.5

CACHE_DIR = "./hotpot_local_index"

INDEX_PATH = os.path.join(
    CACHE_DIR,
    "hotpot_validation.faiss",
)

CORPUS_PATH = os.path.join(
    CACHE_DIR,
    "hotpot_validation_corpus.json",
)


# ============================================================
# GLOBAL RETRIEVAL OBJECTS
# ============================================================

embedder = None
faiss_index = None
corpus = None


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

    for attempt in range(
        OPENROUTER_MAX_RETRIES
    ):
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
                    "model":
                        model,
                    "messages":
                        messages,
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

            if response.status_code == 200:
                data = response.json()

                usage = (
                    data.get("usage", {})
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
                    ["message"]
                    ["content"]
                )

                if DEBUG:
                    print()
                    print("[LLM RESPONSE]")
                    print(content[:2000])

                return content

            if (
                response.status_code == 429
                or
                response.status_code >= 500
            ):
                last_error = (
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                wait = None

                if retry_after:
                    try:
                        wait = float(
                            retry_after
                        )
                    except Exception:
                        pass

                if wait is None:
                    wait = (
                        OPENROUTER_RETRY_BASE
                        *
                        (2 ** attempt)
                    )

                wait += random.uniform(
                    0,
                    0.5,
                )

                print(
                    f"[OPENROUTER RETRY] "
                    f"status={response.status_code} "
                    f"attempt="
                    f"{attempt + 1}/"
                    f"{OPENROUTER_MAX_RETRIES} "
                    f"sleep={wait:.2f}s"
                )

                time.sleep(wait)

                continue

            raise RuntimeError(
                f"OpenRouter error "
                f"{response.status_code}: "
                f"{response.text[:2000]}"
            )

        except requests.RequestException as e:
            last_error = repr(e)

            wait = (
                OPENROUTER_RETRY_BASE
                *
                (2 ** attempt)
                +
                random.uniform(
                    0,
                    0.5,
                )
            )

            print(
                f"[OPENROUTER NETWORK RETRY] "
                f"attempt="
                f"{attempt + 1}/"
                f"{OPENROUTER_MAX_RETRIES} "
                f"sleep={wait:.2f}s"
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
    original = text

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

    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and
        end > start
    ):
        candidate = text[
            start:
            end + 1
        ]

        try:
            return json.loads(
                candidate
            )
        except Exception:
            pass

    print()
    print("[JSON PARSE ERROR]")
    print(original[:4000])

    raise ValueError(
        f"Could not parse JSON:\n"
        f"{original[:4000]}"
    )


# ============================================================
# DATASET
# ============================================================

def load_hotpot_validation():
    print()
    print("=" * 100)
    print("LOAD HOTPOTQA VALIDATION")
    print("=" * 100)

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    print(
        "validation rows:",
        len(ds)
    )

    return ds


def load_hotpot_comparison(
    ds,
    n=100,
    seed=42,
):
    comparison = ds.filter(
        lambda x:
            x["type"]
            ==
            "comparison"
    )

    print(
        "comparison rows:",
        len(comparison)
    )

    comparison = (
        comparison
        .shuffle(seed=seed)
    )

    n = min(
        n,
        len(comparison)
    )

    comparison = (
        comparison
        .select(
            range(n)
        )
    )

    samples = []

    for row in comparison:
        samples.append({
            "id":
                str(
                    row["id"]
                ),

            "question":
                str(
                    row["question"]
                ),

            "answer":
                str(
                    row["answer"]
                ),

            "type":
                str(
                    row["type"]
                ),

            "level":
                str(
                    row["level"]
                ),
        })

    print()
    print(
        f"selected={len(samples)} "
        f"seed={seed}"
    )

    print()
    print("[FIRST 5]")

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

    return samples


# ============================================================
# BUILD LOCAL CORPUS
# ============================================================

def build_corpus(ds):
    docs = {}

    print()
    print("=" * 100)
    print("BUILD DOCUMENT CORPUS")
    print("=" * 100)

    for row in tqdm(
        ds,
        desc="collect corpus",
    ):
        context = row["context"]

        titles = (
            context["title"]
        )

        sentences_list = (
            context["sentences"]
        )

        for title, sentences in zip(
            titles,
            sentences_list,
        ):
            title = str(
                title
            ).strip()

            text = " ".join(
                sentences
            ).strip()

            if not title:
                continue

            if not text:
                continue

            # title 단위 unique corpus
            if title not in docs:
                docs[title] = {
                    "title":
                        title,

                    "text":
                        text,
                }

    result = list(
        docs.values()
    )

    print(
        "unique documents:",
        len(result)
    )

    return result


# ============================================================
# EMBEDDING / FAISS
# ============================================================

def initialize_embedder(
    device=None,
):
    global embedder

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )

    print()
    print("=" * 100)
    print("LOAD EMBEDDING MODEL")
    print("=" * 100)

    print(
        "model :",
        EMBED_MODEL_NAME
    )

    print(
        "device:",
        device
    )

    embedder = (
        SentenceTransformer(
            EMBED_MODEL_NAME,
            device=device,
        )
    )


def build_faiss_index(
    corpus_data,
    batch_size=256,
):
    global faiss_index

    texts = [
        (
            f"{d['title']}. "
            f"{d['text']}"
        )
        for d in corpus_data
    ]

    print()
    print("=" * 100)
    print("EMBED CORPUS")
    print("=" * 100)

    embeddings = (
        embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    embeddings = (
        embeddings.astype(
            "float32"
        )
    )

    dim = (
        embeddings.shape[1]
    )

    print(
        "embedding shape:",
        embeddings.shape
    )

    # cosine similarity
    # normalized embeddings + inner product
    index = (
        faiss.IndexFlatIP(
            dim
        )
    )

    index.add(
        embeddings
    )

    faiss_index = index

    return index


def save_index_and_corpus():
    os.makedirs(
        CACHE_DIR,
        exist_ok=True,
    )

    faiss.write_index(
        faiss_index,
        INDEX_PATH,
    )

    with open(
        CORPUS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            corpus,
            f,
            ensure_ascii=False,
        )

    print()
    print(
        "[INDEX SAVED]",
        INDEX_PATH
    )

    print(
        "[CORPUS SAVED]",
        CORPUS_PATH
    )


def load_cached_index():
    global faiss_index
    global corpus

    if (
        not os.path.exists(
            INDEX_PATH
        )
        or
        not os.path.exists(
            CORPUS_PATH
        )
    ):
        return False

    print()
    print("=" * 100)
    print("LOAD CACHED INDEX")
    print("=" * 100)

    faiss_index = (
        faiss.read_index(
            INDEX_PATH
        )
    )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        corpus = json.load(
            f
        )

    print(
        "index docs:",
        faiss_index.ntotal
    )

    print(
        "corpus docs:",
        len(corpus)
    )

    if (
        faiss_index.ntotal
        !=
        len(corpus)
    ):
        raise RuntimeError(
            "FAISS index and corpus "
            "size mismatch"
        )

    return True


# ============================================================
# LOCAL RETRIEVAL
# ============================================================

def local_search(
    query,
    top_k=TOP_K,
):
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    q = q.astype(
        "float32"
    )

    scores, ids = (
        faiss_index.search(
            q,
            top_k,
        )
    )

    docs = []

    for rank, (
        idx,
        score,
    ) in enumerate(
        zip(
            ids[0],
            scores[0],
        ),
        start=1,
    ):
        if idx < 0:
            continue

        item = (
            corpus[
                int(idx)
            ]
        )

        docs.append({
            "rank":
                rank,

            "title":
                item["title"],

            "text":
                item["text"][
                    :MAX_DOC_CHARS
                ],

            "score":
                float(score),
        })

    return docs


def retrieve(
    query,
    stats,
    top_k=TOP_K,
):
    start = (
        time.perf_counter()
    )

    stats.search_calls += 1

    docs = local_search(
        query,
        top_k=top_k,
    )

    stats.search_latency += (
        time.perf_counter()
        -
        start
    )

    if DEBUG:
        print()
        print(
            "[LOCAL SEARCH]",
            query
        )

        for d in docs:
            print(
                d["rank"],
                f"{d['score']:.4f}",
                d["title"]
            )

    return docs


def docs_to_text(
    query,
    docs,
):
    chunks = [
        f"SEARCH QUERY: {query}"
    ]

    for d in docs:
        chunks.append(
            (
                f"[{d['rank']}] "
                f"{d['title']} "
                f"(score={d['score']:.4f})\n"
                f"{d['text']}"
            )
        )

    return "\n\n".join(
        chunks
    )


# ============================================================
# FINAL ANSWER
# ============================================================

FINAL_ANSWER_PROMPT = """
Answer the original question using only the retrieved evidence.

Rules:
- Give only the shortest correct answer.
- Do not explain your reasoning.
- Do not prefix with "The answer is".
- For yes/no questions output exactly "yes" or "no".
- For comparison questions perform the comparison using the evidence.
- If the answer is a person/entity, output only that entity.
- Do not add unnecessary prose.

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
# ONE-SHOT DECOMPOSITION -> PARALLEL RETRIEVAL
# ============================================================

PARALLEL_DECOMP_PROMPT = """
Decompose this comparison question into independent search queries.

Important:
- Generate ALL required search queries now.
- The queries will be executed simultaneously.
- No query may depend on another search result.
- Usually 2 queries are enough.
- Use 2 to 4 concise search queries.
- Do NOT answer the original question.
- Do NOT create a sequential plan.
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
    # --------------------------------------------------------
    # ONE-SHOT DECOMPOSITION
    # --------------------------------------------------------

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

    plan = extract_json(
        raw
    )

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

    # --------------------------------------------------------
    # TRUE PARALLEL LOCAL RETRIEVAL
    # --------------------------------------------------------

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
                retrieve,
                query,
                stats,
                TOP_K,
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
                futures[
                    future
                ]
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

    return (
        answer,
        {
            "queries":
                queries,

            "parallel_search_wall_latency":
                parallel_search_wall_latency,

            "evidence":
                evidence,
        }
    )


# ============================================================
# METHOD 2
# REACT
# ============================================================

REACT_PROMPT = """
You are answering a question using a local document search tool.

Question:
{question}

Previous search trajectory:
{history}

Choose exactly ONE next action.

If more information is needed:

{{
  "action": "search",
  "query": "..."
}}

If the retrieved evidence is sufficient:

{{
  "action": "finish"
}}

Rules:
- Search exactly one query per turn.
- Use previous observations to decide the next query.
- Do not answer the original question here.
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

    for _ in range(
        MAX_REACT_STEPS
    ):
        history = (
            "\n\n".join(
                trajectory
            )
            if trajectory
            else "(none)"
        )

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
            str(
                action.get(
                    "action",
                    ""
                )
            )
            .strip()
            .lower()
        )

        if action_type == "finish":
            break

        if action_type != "search":
            break

        query = str(
            action.get(
                "query",
                ""
            )
        ).strip()

        if not query:
            break

        if query in queries:
            break

        queries.append(
            query
        )

        docs = retrieve(
            query,
            stats,
            TOP_K,
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

    return (
        answer,

        {
            "queries":
                queries,

            "trajectory":
                trajectory,

            "evidence":
                evidence,
        }
    )


# ============================================================
# METRICS
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
            remove_punc(
                s
            )
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
    pred_tokens = (
        normalize_answer(
            prediction
        )
        .split()
    )

    gold_tokens = (
        normalize_answer(
            gold
        )
        .split()
    )

    if (
        not pred_tokens
        or
        not gold_tokens
    ):
        return float(
            pred_tokens
            ==
            gold_tokens
        )

    common = (
        Counter(pred_tokens)
        &
        Counter(gold_tokens)
    )

    num_same = sum(
        common.values()
    )

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        /
        len(pred_tokens)
    )

    recall = (
        num_same
        /
        len(gold_tokens)
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
# MANIFEST
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
# RUN SINGLE SAMPLE
# ============================================================

def run_one_sample(
    method,
    sample,
    model,
):
    stats = RunStats()

    prediction = ""
    trace = {}

    start = (
        time.perf_counter()
    )

    try:
        if method == "parallel_decomp":
            prediction, trace = (
                run_parallel_decomposition(
                    sample["question"],
                    stats,
                    model,
                )
            )

        elif method == "react":
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

        print(
            error
        )

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

        "normalized_gold":
            normalize_answer(
                sample["answer"]
            ),

        "normalized_prediction":
            normalize_answer(
                prediction
            ),

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
# RUN METHOD
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
            ] = (
                index,
                sample,
            )

        count = 0
        running_em = 0.0
        running_f1 = 0.0
        errors = 0

        pbar = tqdm(
            total=len(samples),
            desc=method,
        )

        for future in as_completed(
            futures
        ):
            (
                index,
                sample,
            ) = futures[
                future
            ]

            try:
                row = future.result()

            except Exception:
                err = (
                    traceback.format_exc()
                )

                row = {
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
                        "",

                    "normalized_gold":
                        normalize_answer(
                            sample["answer"]
                        ),

                    "normalized_prediction":
                        "",

                    "em":
                        0,

                    "f1":
                        0,

                    "wall_latency":
                        0,

                    "llm_calls":
                        0,

                    "search_calls":
                        0,

                    "prompt_tokens":
                        0,

                    "completion_tokens":
                        0,

                    "llm_latency":
                        0,

                    "search_latency":
                        0,

                    "executed_steps":
                        0,

                    "error":
                        err,

                    "trace":
                        "{}",
                }

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

                err=
                    errors,
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
    print("-" * 100)
    print("[RESULT]")
    print("-" * 100)

    print(
        "N              :",
        len(df)
    )

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
        df[
            "search_calls"
        ].mean()
    )

    print(
        "LLM calls      :",
        df[
            "llm_calls"
        ].mean()
    )

    print(
        "executed steps :",
        df[
            "executed_steps"
        ].mean()
    )

    print(
        "wall latency   :",
        df[
            "wall_latency"
        ].mean()
    )

    print(
        "LLM latency    :",
        df[
            "llm_latency"
        ].mean()
    )

    print(
        "search latency :",
        df[
            "search_latency"
        ].mean()
    )

    print(
        "errors         :",
        df[
            "error"
        ].notna().sum()
    )

    print(
        "saved          :",
        path
    )

    return df


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

            "executed_steps":
                df[
                    "executed_steps"
                ].mean(),

            "wall_latency":
                df[
                    "wall_latency"
                ].mean(),

            "llm_latency":
                df[
                    "llm_latency"
                ].mean(),

            "search_latency":
                df[
                    "search_latency"
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
    print("=" * 140)
    print("FINAL SUMMARY")
    print("=" * 140)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "saved:",
        path
    )

    return summary


# ============================================================
# VERIFY SAME EVAL SET
# ============================================================

def verify_same_eval(
    parallel_df,
    react_df,
):
    cols = [
        "id",
        "question",
        "gold",
    ]

    a = (
        parallel_df[
            cols
        ]
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    b = (
        react_df[
            cols
        ]
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    same = a.equals(
        b
    )

    print()
    print("=" * 100)
    print("VERIFY SAME EVAL SET")
    print("=" * 100)

    print(
        "same:",
        same
    )

    print(
        "parallel n:",
        len(a)
    )

    print(
        "react n:",
        len(b)
    )


# ============================================================
# PAIRED ANALYSIS
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
                "question",
                "gold",
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
        validate="one_to_one",
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

    merged[
        "f1_delta"
    ] = (
        merged[
            "parallel_f1"
        ]
        -
        merged[
            "react_f1"
        ]
    )

    merged[
        "latency_speedup"
    ] = (
        merged[
            "react_latency"
        ]
        /
        merged[
            "parallel_latency"
        ].clip(
            lower=1e-9
        )
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
    print("=" * 120)
    print("PAIRED ANALYSIS")
    print("=" * 120)

    print(
        "parallel only correct:",
        int(
            merged[
                "parallel_only_correct"
            ].sum()
        )
    )

    print(
        "react only correct   :",
        int(
            merged[
                "react_only_correct"
            ].sum()
        )
    )

    print(
        "both correct         :",
        int(
            merged[
                "both_correct"
            ].sum()
        )
    )

    print(
        "both wrong           :",
        int(
            merged[
                "both_wrong"
            ].sum()
        )
    )

    print()
    print(
        "parallel F1:",
        merged[
            "parallel_f1"
        ].mean()
    )

    print(
        "react F1   :",
        merged[
            "react_f1"
        ].mean()
    )

    print()
    print(
        "parallel latency:",
        merged[
            "parallel_latency"
        ].mean()
    )

    print(
        "react latency   :",
        merged[
            "react_latency"
        ].mean()
    )

    print(
        "react / parallel speedup:",
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

    # --------------------------------------------------------
    # PARALLEL-ONLY EXAMPLES
    # --------------------------------------------------------

    wins = (
        merged[
            merged[
                "parallel_only_correct"
            ]
        ]
    )

    if len(wins):
        print()
        print("=" * 120)
        print("PARALLEL ONLY CORRECT EXAMPLES")
        print("=" * 120)

        for _, row in (
            wins.head(10)
            .iterrows()
        ):
            print()
            print(
                "Q:",
                row["question"]
            )

            print(
                "GOLD:",
                row["gold"]
            )

            print(
                "PARALLEL:",
                row[
                    "parallel_prediction"
                ]
            )

            print(
                "REACT:",
                row[
                    "react_prediction"
                ]
            )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    global corpus

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
            "./parallel_vs_react_local_results",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
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
        "LOCAL FAISS "
        "PARALLEL DECOMPOSITION "
        "VS REACT"
    )

    print("=" * 120)

    print(
        "LLM model   :",
        args.model
    )

    print(
        "embed model :",
        EMBED_MODEL_NAME
    )

    print(
        "n           :",
        args.n
    )

    print(
        "seed        :",
        args.seed
    )

    print(
        "workers     :",
        args.workers
    )

    print(
        "top-k       :",
        TOP_K
    )

    print(
        "reasoning   :",
        False
    )

    # ========================================================
    # DATA
    # ========================================================

    ds = (
        load_hotpot_validation()
    )

    samples = (
        load_hotpot_comparison(
            ds=ds,
            n=args.n,
            seed=args.seed,
        )
    )

    save_manifest(
        samples,
        args.output_dir,
    )

    # ========================================================
    # EMBEDDER
    # ========================================================

    initialize_embedder()

    # ========================================================
    # INDEX
    # ========================================================

    cache_ok = (
        load_cached_index()
        if not args.rebuild_index
        else False
    )

    if not cache_ok:
        corpus = build_corpus(
            ds
        )

        build_faiss_index(
            corpus
        )

        save_index_and_corpus()

    # ========================================================
    # RUN
    # ========================================================

    dfs = {}

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
        verify_same_eval(
            dfs[
                "parallel_decomp"
            ],

            dfs[
                "react"
            ],
        )

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
