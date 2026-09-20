# LLM Testing Project

Automated testing project for evaluating a local Large Language Model using LM Studio and DeepEval.
The project evaluates LLM responses across five quality dimensions:
* Accuracy and Correctness
* Answer Relevancy
* Hallucination Detection
* Toxicity
* Bias
The project contains 13 automated test cases and uses five DeepEval metrics.

## Project Structure
llm_testing_project/
│
├── tests/
│   ├── test_accuracy.py
│   ├── test_relevancy.py
│   ├── test_hallucination.py
│   ├── test_toxicity.py
│   └── test_bias.py
│
├── helpers/
│   ├── llm_client.py
│   └── test_utils.py
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── .gitignore
├── requirements.txt
└── README.md

## Technologies
* Python
* LM Studio
* Llama 3.1 8B Instruct
* DeepEval
* pytest
* GitHub Actions

## Requirements
* Python 3.12 or newer
* LM Studio or another OpenAI-compatible LLM server
* A loaded instruction-following LLM
* DeepEval 4.2.2

## Local LLM Configuration
The project expects an OpenAI-compatible API endpoint.

The default local configuration is:
Base URL:
http://127.0.0.1:1234
Model:
meta-llama-3.1-8b-instruct

LM Studio should have the selected model loaded and its local server enabled.

The application sends requests to:
http://127.0.0.1:1234/v1/chat/completions

## Installation
Create and activate a virtual environment.

On Windows:
python -m venv .venv
.venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt

Verify the installation:
deepeval --version

Expected version:
4.2.2

## Running the Tests Locally
Make sure LM Studio is running and the model is loaded.
Then run:
deepeval test run tests/

You can also run an individual test file:
deepeval test run tests/test_accuracy.py
deepeval test run tests/test_relevancy.py
deepeval test run tests/test_hallucination.py
deepeval test run tests/test_toxicity.py
deepeval test run tests/test_bias.py

## Test Coverage
### Accuracy
File:
tests/test_accuracy.py

Metric:
GEval

The tests verify whether the generated answer is factually correct compared with the expected reference answer.
Threshold:
0.7

### Answer Relevancy
File:
tests/test_relevancy.py

Metric:
AnswerRelevancyMetric

The metric checks whether the generated response directly addresses the user's question.
Threshold:
0.7

### Hallucination Detection
File:
tests/test_hallucination.py

Metric:
HallucinationMetric

The tests provide trusted context and verify whether the generated response contradicts or invents information that is not supported by the context.
Threshold:
0.8

### Toxicity
File:
tests/test_toxicity.py

Metric:
ToxicityMetric

The tests verify that generated responses remain appropriate and avoid toxic language.
Threshold:
0.8

### Bias
File:
tests/test_bias.py

Metric:
BiasMetric

The tests verify that responses avoid gender, racial, geographical, and other forms of biased opinions.
Threshold:
0.8

## Test Case Design
Each test case contains the information required by its evaluation metric.
A typical test case contains:
LLMTestCase(
    input=question,
    actual_output=generated_output,
    expected_output=expected_output,
    context=context,
)

The `actual_output` is generated dynamically by the configured LLM.
Reference information is provided only when required by the metric.

## Environment Variables
The LLM connection can be configured using environment variables.
LLM_BASE_URL
LLM_MODEL
LLM_API_KEY

If these variables are not provided, the application uses the local LM Studio defaults:
LLM_BASE_URL=http://127.0.0.1:1234
LLM_MODEL=meta-llama-3.1-8b-instruct

## GitHub Actions
The project contains a GitHub Actions workflow:
.github/workflows/test.yml

The workflow runs automatically on:
* push to `main`
* push to `master`
* pull requests
The workflow installs project dependencies and executes:
deepeval test run tests/
