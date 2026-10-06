# eval_retriever.py
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = OpenAIModel(model="google/gemma-4-31b-it:free",
                          base_url="https://www.zerolimitai.com/api/v1",
                          api_key=os.getenv("ZEROLIMIT_API_KEY"),
                          temperature=0)
THRESHOLD = 0.7


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",          # vs "reranked" when you swap it in
            "embedding_model": "qwen3-embedding:4b",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 5,
            "judge_model": "google/gemma-4-31b-it:free",
            "golden_set": GOLDEN_PATH,
        },
        async_config=AsyncConfig(run_async=True)
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever(fetch_k=20, top_k=5))


if __name__ == "__main__":
    print_summary("retriever", run_local())