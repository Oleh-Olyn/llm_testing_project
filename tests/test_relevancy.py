import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

from helpers.llm_client import LocalLLM
from helpers.test_utils import generate_test_case


local_llm = LocalLLM()

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=local_llm,
    async_mode=False,
)


TEST_CASES = [
    "What is Python used for?",
    "What is the purpose of a database?",
]


@pytest.mark.parametrize("question", TEST_CASES)
def test_answer_relevancy(question):
    """
    Preconditions:
    - LM Studio is running.
    - The local model is available.

    Steps:
    1. Generate an answer dynamically using the local LLM.
    2. Evaluate the answer with AnswerRelevancyMetric.

    Expected result:
    - The answer directly addresses the user's question.
    - Relevancy score is at least 0.7.
    """

    test_case = generate_test_case(question)

    assert_test(
        test_case=test_case,
        metrics=[relevancy_metric],
        run_async=False,
    )