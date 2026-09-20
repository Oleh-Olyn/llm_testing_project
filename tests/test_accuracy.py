import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from helpers.test_utils import generate_test_case, local_llm


accuracy_metric = GEval(
    name="Accuracy",
    evaluation_steps=[
        "Check whether the actual output contains factually correct information.",
        "Compare the actual output with the expected output.",
        "Penalize factual contradictions and incorrect claims.",
        "Penalize important missing information.",
        "Do not penalize harmless differences in wording.",
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=local_llm,
)


TEST_CASES = [
    {
        "question": "What is the capital of France?",
        "expected": "Paris is the capital of France.",
    },
    {
        "question": "What is 2 + 2?",
        "expected": "2 + 2 equals 4.",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "expected": "William Shakespeare wrote Romeo and Juliet.",
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_accuracy(case):
    test_case = generate_test_case(
        question=case["question"],
        expected_output=case["expected"],
    )

    assert_test(
        test_case=test_case,
        metrics=[accuracy_metric],
    )