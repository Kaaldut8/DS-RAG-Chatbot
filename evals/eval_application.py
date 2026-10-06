# eval_application.py

import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL = OpenAIModel(model="google/gemma-4-31b-it:free",
                          base_url="https://www.zerolimitai.com/api/v1",
                          api_key=os.getenv("ZEROLIMIT_API_KEY"),
                          temperature=0)
THRESHOLD = 0.7


def run(rag):

    # ---------------------------------------------------------
    # 1. LOAD GOLDEN DATASET
    # ---------------------------------------------------------

    goldens = load_goldens(GOLDEN_PATH)

    # ---------------------------------------------------------
    # 2. RUN LIVE DS-RAG PIPELINE
    # ---------------------------------------------------------

    test_cases = []

    for g in goldens:

        result = rag.invoke(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"],
            )
        )

    # ---------------------------------------------------------
    # 3. APPLICATION-LEVEL GEVALS
    # ---------------------------------------------------------

    # =========================================================
    # CORRECTNESS
    # =========================================================

    correctness = GEval(
        name="Correctness",

        evaluation_steps=[
            "Read the question, expected answer, and actual answer.",
            "Identify the factual claims made in the actual answer.",
            "Compare those claims against the expected answer.",
            "A claim should be considered incorrect if it contradicts the expected answer or contains a clear factual error.",
            "Judge factual correctness rather than wording or exact phrase matching.",
            "Do not penalize the answer merely because it is shorter than the expected answer. Missing information is evaluated by the Completeness metric.",
            "Do not penalize additional information when that information is factually correct and does not contradict the expected answer.",
            "For Data Science concepts, pay particular attention to definitions, formulas, relationships between concepts, algorithm behavior, and technical terminology.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "The answer is factually correct and contains no "
                    "meaningful contradictions or technical errors."
                ),
            ),

            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "The answer is mostly correct but contains a minor "
                    "technical inaccuracy or imprecise statement."
                ),
            ),

            Rubric(
                score_range=(4, 6),
                expected_outcome=(
                    "The answer contains some correct information but also "
                    "contains one or more significant inaccuracies."
                ),
            ),

            Rubric(
                score_range=(0, 3),
                expected_outcome=(
                    "The answer contains major factual errors, incorrect "
                    "definitions, incorrect formulas, or contradictions "
                    "with the expected answer."
                ),
            ),
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # =========================================================
    # COMPLETENESS
    # =========================================================

    completeness = GEval(
        name="Completeness",

        evaluation_steps=[
            "Read the question and expected answer.",
            "Identify the important concepts, facts, relationships, steps, or formulas that the expected answer contains.",
            "Check whether the actual answer covers those important points.",
            "Give credit when the actual answer expresses the same concept using different wording.",
            "Penalize the answer when it omits important concepts required to properly answer the question.",
            "Do not require the actual answer to reproduce every sentence or detail from the expected answer.",
            "Do not penalize additional information that is relevant to the question.",
            "Judge coverage only. Do not penalize factual errors here; those are evaluated separately by Correctness.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "The answer covers essentially all important concepts "
                    "required by the expected answer."
                ),
            ),

            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "The answer covers most important concepts but misses "
                    "one relatively minor point."
                ),
            ),

            Rubric(
                score_range=(4, 6),
                expected_outcome=(
                    "The answer covers some important concepts but misses "
                    "one or more significant points."
                ),
            ),

            Rubric(
                score_range=(0, 3),
                expected_outcome=(
                    "The answer covers very little of the expected content "
                    "or fails to address important parts of the question."
                ),
            ),
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # =========================================================
    # STYLE
    # =========================================================

    style = GEval(
        name="Style",

        evaluation_steps=[
            "Judge only the quality of the explanation and teaching style.",
            "The answer should directly answer the Data Science question without unnecessary discussion.",
            "Prefer clear, simple language that makes technical concepts easy to understand.",
            "Technical terms such as embeddings, regularization, precision, recall, latent space, attention, or reranking should be explained naturally when the target audience may not already understand them.",
            "Prefer an intuitive explanation before introducing formulas or highly technical details when appropriate.",
            "The answer should feel like a knowledgeable Data Science instructor explaining a concept to a learner.",
            "A concise example or analogy is useful for abstract concepts, but an answer must not be penalized for omitting an analogy when the explanation is already clear.",
            "Penalize unnecessary verbosity, repetition, excessive sectioning, unexplained jargon, or robotic wording.",
            "Do not require bullet points or prose specifically. Use whatever structure makes the explanation clearest.",
            "Do not judge factual correctness or completeness in this metric.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Clear, intuitive, concise, and natural Data Science "
                    "teaching style. The explanation is easy to follow "
                    "and appropriately technical for the question."
                ),
            ),

            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "Clear and useful explanation with only minor issues "
                    "such as slightly formal wording, verbosity, or jargon."
                ),
            ),

            Rubric(
                score_range=(4, 6),
                expected_outcome=(
                    "Understandable but somewhat confusing, overly formal, "
                    "verbose, repetitive, or jargon-heavy."
                ),
            ),

            Rubric(
                score_range=(0, 3),
                expected_outcome=(
                    "Very difficult to follow, robotic, excessively verbose, "
                    "poorly structured, or filled with unexplained jargon."
                ),
            ),
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # ---------------------------------------------------------
    # 4. RUN EVALUATION
    # ---------------------------------------------------------

    result = evaluate(
        test_cases=test_cases,
        metrics=[
            correctness,
            completeness,
            style,
        ],
    )

    return summarize_by_metric(result)


def run_local():
    """Build the DS-RAG pipeline and evaluate it."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())