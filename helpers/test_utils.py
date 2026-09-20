from deepeval.test_case import LLMTestCase

from helpers.llm_client import LocalLLM


local_llm = LocalLLM()


def generate_test_case(
    question: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
) -> LLMTestCase:

    if context:
        context_text = "\n".join(
            f"- {item}" for item in context
        )

        prompt = f"""
You are answering a factual evaluation question.

Use ONLY the information in the provided context.

Context:
{context_text}

Question:
{question}

Important instructions:
- Return a normal natural-language answer.
- Do NOT use tools.
- Do NOT return JSON.
- Do NOT return a function call.
- Do NOT return fields such as "name", "parameters", "arguments", or "tool_calls".
- Answer directly and concisely.
""".strip()

    else:
        prompt = f"""
Answer the following question directly.

Question:
{question}

Important instructions:
- Return a normal natural-language answer.
- Do NOT use tools.
- Do NOT return JSON.
- Do NOT return a function call.
- Do NOT return fields such as "name", "parameters", "arguments", or "tool_calls".
- Answer directly and concisely.
""".strip()

    actual_output = local_llm.generate(prompt)

    return LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
    )