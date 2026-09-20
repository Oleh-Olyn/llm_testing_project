import pytest

from deepeval import assert_test
from deepeval.metrics import HallucinationMetric

from helpers.llm_client import LocalLLM
from helpers.test_utils import generate_test_case


local_llm = LocalLLM()

hallucination_metric = HallucinationMetric(
    threshold=0.8,
    model=local_llm,
    async_mode=False,
)


TEST_CASES = [
    {
        "question": "Where is the Eiffel Tower located?",
        "context": [
            "The Eiffel Tower is located in Paris, France.",
        ],
    },
    {
        "question": "At what temperature does water freeze at standard atmospheric pressure?",
        "context": [
            "At standard atmospheric pressure, water freezes at 0 degrees Celsius.",
        ],
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_hallucination(case):
    """
    Preconditions:
    - LM Studio is running.
    - The local model is available.

    Steps:
    1. Generate an answer using only the supplied context.
    2. Evaluate the answer with HallucinationMetric.

    Expected result:
    - The answer should be supported by the provided context.
    - Hallucination score should be at least 0.8.
    """

    test_case = generate_test_case(
        question=case["question"],
        context=case["context"],
    )

    assert_test(
        test_case=test_case,
        metrics=[hallucination_metric],
        run_async=False,
    )