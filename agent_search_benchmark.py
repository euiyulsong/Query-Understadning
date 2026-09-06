import os
import re
import json
import time
import random
import string
import argparse
import threading
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
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "google/gemini-2.5-flash"

DATASETS = [
    "hotpotqa",
    "2wiki",
    "musique",
    "popqa",
]

METHODS = [
    "parallel_plan",
    "plan_execute",
    "plan_replan",
    "react",
]

MAX_SEARCH_RESULTS = 5
MAX_AGENT_STEPS = 6

# Search snippets can get long, so keep them bounded.
MAX_DOC_CHARS = 1800

SEED = 42

# Number of concurrent independent search calls.
SEARCH_WORKERS = 8

HF_DATASET = "AQ-MedAI/RAG-QA-Leaderboard"


# ============================================================
# THREAD LOCAL HTTP SESSION
# ============================================================

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": "agent-search-benchmark/0.1"
        })
        _local.session = s
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

        self.search_latency = 0.0
        self.llm_latency = 0.0

        self.replans = 0
        self.executed_steps = 0

    def to_dict(self):
        return vars(self).copy()


# ============================================================
# OPENROUTER
# ============================================================

def call_llm(
    messages,
    stats: RunStats,
    model=DEFAULT_MODEL,
    temperature=0.0,
    max_tokens=1000,
):
    start = time.perf_counter()

    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )

    latency = time.perf_counter() - start

    stats.llm_calls += 1
    stats.llm_latency += latency

    if r.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {r.status_code}: {r.text[:1000]}"
        )

    data = r.json()

    usage = data.get("usage", {}) or {}

    stats.prompt_tokens += usage.get("prompt_tokens", 0) or 0
    stats.completion_tokens += usage.get("completion_tokens", 0) or 0

    return data["choices"][0]["message"]["content"]


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):
    text = text.strip()

    # ```json ... ```
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Find object / list in prose
    candidates = []

    obj_start = text.find("{")
    obj_end = text.rfind("}")

    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])

    arr_start = text.find("[")
    arr_end = text.rfind("]")

    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end + 1])

    for x in candidates:
        try:
            return json.loads(x)
        except Exception:
            continue

    raise ValueError(f"Could not parse JSON:\n{text[:2000]}")


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

WIKI_API = "https://en.wikipedia.org/w/api.php"


def wikipedia_search(query, stats, top_k=MAX_SEARCH_RESULTS):
    """
    Same retrieval backend for all agent strategies.

    1. MediaWiki full-text search
    2. Fetch page extracts for top results
    """

    start = time.perf_counter()
    stats.search_calls += 1

    s = get_session()

    try:
        r = s.get(
            WIKI_API,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": top_k,
                "utf8": 1,
            },
            timeout=30,
        )
        r.raise_for_status()

        hits = r.json().get("query", {}).get("search", [])

        if not hits:
            return []

        titles = [x["title"] for x in hits]

        r2 = s.get(
            WIKI_API,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 0,
                "explaintext": 1,
                "redirects": 1,
                "titles": "|".join(titles),
            },
            timeout=30,
        )

        r2.raise_for_status()

        pages = r2.json().get("query", {}).get("pages", {})

        title_to_extract = {}

        for page in pages.values():
            title = page.get("title", "")
            extract = page.get("extract", "")
            title_to_extract[title.lower()] = extract

        docs = []

        for rank, hit in enumerate(hits, start=1):
            title = hit["title"]

            extract = title_to_extract.get(title.lower(), "")

            if not extract:
                # MediaWiki search snippet fallback
                snippet = re.sub("<.*?>", " ", hit.get("snippet", ""))
                extract = snippet

            docs.append({
                "rank": rank,
                "title": title,
                "text": extract[:MAX_DOC_CHARS],
            })

        return docs

    except Exception as e:
        return [{
            "rank": 1,
            "title": "SEARCH_ERROR",
            "text": str(e),
        }]

    finally:
        stats.search_latency += time.perf_counter() - start


def docs_to_text(query, docs):
    chunks = [f"SEARCH QUERY: {query}"]

    for d in docs:
        chunks.append(
            f"""
[{d['rank']}] {d['title']}
{d['text']}
""".strip()
        )

    return "\n\n".join(chunks)


# ============================================================
# ANSWER GENERATION
# ============================================================

ANSWER_PROMPT = """
Answer the question using only the retrieved evidence.

Rules:
- Give the shortest answer that correctly answers the question.
- Do not explain your reasoning.
- Do not include phrases such as "The answer is".
- For a person, return the person's name.
- For a date, return the date.
- For yes/no questions, return "yes" or "no".
- If comparison is required, perform the comparison yourself.

Question:
{question}

Evidence:
{evidence}
"""


def final_answer(question, evidence, stats, model):
    return call_llm(
        [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded question answering system."
                ),
            },
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    question=question,
                    evidence=evidence,
                ),
            },
        ],
        stats,
        model=model,
        max_tokens=200,
    ).strip()


# ============================================================
# 1. PARALLEL PLAN
# ============================================================

PARALLEL_PLAN_PROMPT = """
You are planning web searches for a question.

Create independent searches that can be executed immediately in parallel.

Important:
- Decompose the question.
- Only create searches that do NOT depend on another search result.
- Each search query should be understandable by Wikipedia search.
- Generate 1-5 queries.
- Do not answer the question.

Return JSON only:

{{
  "queries": [
    "query 1",
    "query 2"
  ]
}}

Question:
{question}
"""


def run_parallel_plan(question, stats, model):
    plan_raw = call_llm(
        [
            {
                "role": "user",
                "content": PARALLEL_PLAN_PROMPT.format(
                    question=question
                ),
            }
        ],
        stats,
        model=model,
        max_tokens=500,
    )

    plan = extract_json(plan_raw)

    queries = plan.get("queries", [])

    queries = [
        str(q).strip()
        for q in queries
        if str(q).strip()
    ][:5]

    if not queries:
        queries = [question]

    stats.executed_steps += len(queries)

    evidence_parts = []

    # Actual parallel execution
    with ThreadPoolExecutor(
        max_workers=min(SEARCH_WORKERS, len(queries))
    ) as executor:

        futures = {
            executor.submit(
                wikipedia_search,
                q,
                stats,
            ): q
            for q in queries
        }

        for future in as_completed(futures):
            q = futures[future]

            try:
                docs = future.result()
            except Exception as e:
                docs = [{
                    "rank": 1,
                    "title": "ERROR",
                    "text": str(e),
                }]

            evidence_parts.append(
                docs_to_text(q, docs)
            )

    evidence = "\n\n====================\n\n".join(
        evidence_parts
    )

    answer = final_answer(
        question,
        evidence,
        stats,
        model,
    )

    return answer, {
        "initial_plan": plan,
        "evidence": evidence,
    }


# ============================================================
# 2. PLAN -> EXECUTE
# ============================================================

PLAN_EXECUTE_PROMPT = """
Create a search plan for answering the question.

A later step may depend on information found in an earlier step.

Represent variables using {{variable_name}}.

Example:

Question:
Which was born first, the director of Alien or the director of Jaws?

Output:
{{
  "steps": [
    {{
      "id": 1,
      "query": "Alien film director",
      "save_as": "alien_director"
    }},
    {{
      "id": 2,
      "query": "{{alien_director}} date of birth",
      "save_as": "alien_birth"
    }},
    {{
      "id": 3,
      "query": "Jaws film director",
      "save_as": "jaws_director"
    }},
    {{
      "id": 4,
      "query": "{{jaws_director}} date of birth",
      "save_as": "jaws_birth"
    }}
  ]
}}

Rules:
- 1 to 6 steps.
- Keep queries concise.
- Do not answer the question.
- Return JSON only.

Question:
{question}
"""


EXTRACT_VARIABLE_PROMPT = """
A search was executed.

Search query:
{query}

Search results:
{evidence}

Extract the single entity/value that should be saved as:

{variable}

Return only the value.
"""


def substitute_variables(text, memory):
    for key, value in memory.items():
        text = text.replace(
            "{" + key + "}",
            value,
        )

    return text


def run_plan_execute(question, stats, model):
    raw = call_llm(
        [
            {
                "role": "user",
                "content": PLAN_EXECUTE_PROMPT.format(
                    question=question
                ),
            }
        ],
        stats,
        model=model,
        max_tokens=700,
    )

    plan = extract_json(raw)

    steps = plan.get("steps", [])[:MAX_AGENT_STEPS]

    memory = {}
    evidence_parts = []

    for step in steps:
        query_template = str(
            step.get("query", "")
        ).strip()

        if not query_template:
            continue

        query = substitute_variables(
            query_template,
            memory,
        )

        docs = wikipedia_search(
            query,
            stats,
        )

        evidence = docs_to_text(
            query,
            docs,
        )

        evidence_parts.append(evidence)

        stats.executed_steps += 1

        save_as = step.get("save_as")

        if save_as:
            value = call_llm(
                [
                    {
                        "role": "user",
                        "content":
                            EXTRACT_VARIABLE_PROMPT.format(
                                query=query,
                                evidence=evidence,
                                variable=save_as,
                            ),
                    }
                ],
                stats,
                model=model,
                max_tokens=100,
            ).strip()

            memory[save_as] = value

    all_evidence = (
        "\n\n====================\n\n".join(
            evidence_parts
        )
    )

    answer = final_answer(
        question,
        all_evidence,
        stats,
        model,
    )

    return answer, {
        "plan": plan,
        "memory": memory,
        "evidence": all_evidence,
    }


# ============================================================
# 3. PLAN -> EXECUTE -> REPLAN
# ============================================================

INITIAL_REPLAN_PROMPT = """
Generate the FIRST search needed to answer this question.

Do not try to generate the entire final answer.

Return JSON only:

{{
  "query": "...",
  "reason": "..."
}}

Question:
{question}
"""


REPLAN_PROMPT = """
You are an adaptive search planner.

Question:
{question}

Search history:
{history}

Retrieved evidence:
{evidence}

Decide whether enough evidence exists to answer the question.

If enough:

{{
  "done": true,
  "next_query": null
}}

Otherwise:

{{
  "done": false,
  "next_query": "the single best next search"
}}

Rules:
- The next query should use entities discovered from prior searches when helpful.
- Avoid duplicate searches.
- Return JSON only.
"""


def run_plan_replan(question, stats, model):
    first_raw = call_llm(
        [
            {
                "role": "user",
                "content": INITIAL_REPLAN_PROMPT.format(
                    question=question
                ),
            }
        ],
        stats,
        model=model,
        max_tokens=300,
    )

    first = extract_json(first_raw)

    query = first.get("query") or question

    history = []
    evidence_parts = []

    for step_idx in range(MAX_AGENT_STEPS):

        docs = wikipedia_search(
            query,
            stats,
        )

        ev = docs_to_text(
            query,
            docs,
        )

        evidence_parts.append(ev)
        history.append(query)

        stats.executed_steps += 1

        all_evidence = (
            "\n\n====================\n\n".join(
                evidence_parts
            )
        )

        if step_idx == MAX_AGENT_STEPS - 1:
            break

        raw = call_llm(
            [
                {
                    "role": "user",
                    "content": REPLAN_PROMPT.format(
                        question=question,
                        history=json.dumps(
                            history,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        evidence=all_evidence,
                    ),
                }
            ],
            stats,
            model=model,
            max_tokens=300,
        )

        decision = extract_json(raw)

        stats.replans += 1

        if decision.get("done", False):
            break

        next_query = decision.get("next_query")

        if not next_query:
            break

        next_query = str(next_query).strip()

        if next_query in history:
            break

        query = next_query

    all_evidence = (
        "\n\n====================\n\n".join(
            evidence_parts
        )
    )

    answer = final_answer(
        question,
        all_evidence,
        stats,
        model,
    )

    return answer, {
        "first_plan": first,
        "history": history,
        "evidence": all_evidence,
    }


# ============================================================
# 4. REACT
# ============================================================

REACT_PROMPT = """
You are solving an open-domain question using Wikipedia search.

Question:
{question}

Previous observations:
{history}

Choose ONE action.

If more information is required:

{{
  "action": "search",
  "query": "..."
}}

If enough information exists:

{{
  "action": "finish"
}}

Rules:
- Search only one query per turn.
- Use observations from earlier turns.
- Do not hallucinate search results.
- Return JSON only.
"""


def run_react(question, stats, model):
    trajectory = []
    evidence_parts = []

    for _ in range(MAX_AGENT_STEPS):

        history = "\n\n".join(
            trajectory
        ) if trajectory else "(none)"

        raw = call_llm(
            [
                {
                    "role": "user",
                    "content": REACT_PROMPT.format(
                        question=question,
                        history=history,
                    ),
                }
            ],
            stats,
            model=model,
            max_tokens=300,
        )

        action = extract_json(raw)

        if action.get("action") == "finish":
            break

        query = action.get("query")

        if not query:
            break

        query = str(query).strip()

        docs = wikipedia_search(
            query,
            stats,
        )

        ev = docs_to_text(
            query,
            docs,
        )

        evidence_parts.append(ev)

        trajectory.append(
            f"""
ACTION:
search({query})

OBSERVATION:
{ev}
""".strip()
        )

        stats.executed_steps += 1

    all_evidence = (
        "\n\n====================\n\n".join(
            evidence_parts
        )
    )

    answer = final_answer(
        question,
        all_evidence,
        stats,
        model,
    )

    return answer, {
        "trajectory": trajectory,
        "evidence": all_evidence,
    }


# ============================================================
# METRICS
# HotpotQA style EM/F1
# ============================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    def lower(text):
        return text.lower()

    s = unicodedata.normalize(
        "NFD",
        str(s),
    )

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(s)
            )
        )
    )


def exact_match(prediction, ground_truth):
    return int(
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(
        prediction
    ).split()

    gold_tokens = normalize_answer(
        ground_truth
    ).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# DATASET
# ============================================================

def first_existing(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]

    return None


def normalize_row(row, idx):
    q = first_existing(
        row,
        [
            "question",
            "query",
            "input",
        ],
    )

    answer = first_existing(
        row,
        [
            "answer",
            "answers",
            "gold_answer",
            "output",
        ],
    )

    if isinstance(answer, dict):
        # Common HF format:
        # {"text": ["Paris"], ...}
        if "text" in answer:
            answer = answer["text"]

    if isinstance(answer, (list, tuple)):
        if len(answer):
            answer = answer[0]
        else:
            answer = ""

    return {
        "id": str(
            first_existing(
                row,
                ["id", "_id", "qid"],
            )
            or idx
        ),
        "question": str(q),
        "answer": str(answer),
    }


def load_samples(
    dataset_name,
    n=100,
    seed=SEED,
):
    print(
        f"\n[LOAD] {dataset_name}"
    )

    ds = load_dataset(
        HF_DATASET,
        dataset_name,
        split="train",
    )

    # Deterministic experiment sample.
    ds = ds.shuffle(seed=seed)

    result = []

    for i, row in enumerate(ds):
        x = normalize_row(
            row,
            i,
        )

        if (
            x["question"]
            and x["question"] != "None"
            and x["answer"]
            and x["answer"] != "None"
        ):
            result.append(x)

        if len(result) >= n:
            break

    if len(result) < n:
        raise RuntimeError(
            f"{dataset_name}: only found "
            f"{len(result)} usable rows"
        )

    return result


# ============================================================
# METHOD ROUTER
# ============================================================

def run_method(
    method,
    question,
    model,
):
    stats = RunStats()

    start = time.perf_counter()

    if method == "parallel_plan":
        answer, trace = run_parallel_plan(
            question,
            stats,
            model,
        )

    elif method == "plan_execute":
        answer, trace = run_plan_execute(
            question,
            stats,
            model,
        )

    elif method == "plan_replan":
        answer, trace = run_plan_replan(
            question,
            stats,
            model,
        )

    elif method == "react":
        answer, trace = run_react(
            question,
            stats,
            model,
        )

    else:
        raise ValueError(method)

    wall_latency = (
        time.perf_counter() - start
    )

    return answer, trace, stats, wall_latency


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    datasets,
    methods,
    n,
    model,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    all_rows = []

    for dataset_name in datasets:

        samples = load_samples(
            dataset_name,
            n=n,
        )

        for method in methods:

            print()
            print("=" * 100)
            print(
                f"DATASET={dataset_name} "
                f"METHOD={method} "
                f"N={n}"
            )
            print("=" * 100)

            rows = []

            for sample in tqdm(
                samples,
                desc=f"{dataset_name}/{method}",
            ):
                try:
                    (
                        prediction,
                        trace,
                        stats,
                        latency,
                    ) = run_method(
                        method,
                        sample["question"],
                        model,
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

                except Exception as e:
                    prediction = ""
                    trace = {}
                    stats = RunStats()
                    latency = 0

                    em = 0
                    f1 = 0

                    error = repr(e)

                row = {
                    "dataset": dataset_name,
                    "method": method,
                    "id": sample["id"],
                    "question": sample["question"],
                    "gold": sample["answer"],
                    "prediction": prediction,

                    "em": em,
                    "f1": f1,

                    "wall_latency": latency,

                    **stats.to_dict(),

                    "error": error,

                    "trace": json.dumps(
                        trace,
                        ensure_ascii=False,
                    ),
                }

                rows.append(row)
                all_rows.append(row)

            pd.DataFrame(rows).to_csv(
                os.path.join(
                    output_dir,
                    f"{dataset_name}__{method}.csv",
                ),
                index=False,
            )

            # Immediate result
            tmp = pd.DataFrame(rows)

            print()
            print(
                tmp[
                    [
                        "em",
                        "f1",
                        "search_calls",
                        "llm_calls",
                        "wall_latency",
                        "replans",
                    ]
                ].mean()
            )

    df = pd.DataFrame(all_rows)

    df.to_csv(
        os.path.join(
            output_dir,
            "details.csv",
        ),
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        df
        .groupby(
            [
                "dataset",
                "method",
            ],
            as_index=False,
        )
        .agg(
            n=("id", "count"),
            EM=("em", "mean"),
            F1=("f1", "mean"),

            search_calls=(
                "search_calls",
                "mean",
            ),

            llm_calls=(
                "llm_calls",
                "mean",
            ),

            executed_steps=(
                "executed_steps",
                "mean",
            ),

            replans=(
                "replans",
                "mean",
            ),

            latency=(
                "wall_latency",
                "mean",
            ),

            search_latency=(
                "search_latency",
                "mean",
            ),

            llm_latency=(
                "llm_latency",
                "mean",
            ),

            prompt_tokens=(
                "prompt_tokens",
                "mean",
            ),

            completion_tokens=(
                "completion_tokens",
                "mean",
            ),
        )
    )

    summary["f1_per_search"] = (
        summary["F1"]
        / summary["search_calls"].clip(
            lower=1e-9
        )
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "summary.csv",
        ),
        index=False,
    )

    # Overall average across datasets
    overall = (
        df
        .groupby(
            "method",
            as_index=False,
        )
        .agg(
            n=("id", "count"),
            EM=("em", "mean"),
            F1=("f1", "mean"),

            search_calls=(
                "search_calls",
                "mean",
            ),

            llm_calls=(
                "llm_calls",
                "mean",
            ),

            latency=(
                "wall_latency",
                "mean",
            ),

            replans=(
                "replans",
                "mean",
            ),

            prompt_tokens=(
                "prompt_tokens",
                "mean",
            ),

            completion_tokens=(
                "completion_tokens",
                "mean",
            ),
        )
    )

    overall["f1_per_search"] = (
        overall["F1"]
        / overall["search_calls"].clip(
            lower=1e-9
        )
    )

    overall.to_csv(
        os.path.join(
            output_dir,
            "overall.csv",
        ),
        index=False,
    )

    print()
    print("=" * 120)
    print("PER-DATASET RESULT")
    print("=" * 120)
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("OVERALL RESULT")
    print("=" * 120)
    print(
        overall.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
    )

    parser.add_argument(
        "--output-dir",
        default="./agent_search_results",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("AGENT SEARCH ORCHESTRATION BENCHMARK")
    print("=" * 100)

    print("model   :", args.model)
    print("n       :", args.n)
    print("datasets:", args.datasets)
    print("methods :", args.methods)

    benchmark(
        datasets=args.datasets,
        methods=args.methods,
        n=args.n,
        model=args.model,
        output_dir=args.output_dir,
    )
