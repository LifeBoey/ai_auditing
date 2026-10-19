import os
import pandas as pd
from anthropic import Anthropic
from phoenix.otel import register
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from phoenix.evals import LLM, create_classifier, evaluate_dataframe

BASE  = os.environ["YOUR_BASE_URL_VAR"].rstrip("/")
TOKEN = os.environ["YOUR_AUTH_TOKEN_VAR"]
MODEL = os.environ["YOUR_MODEL_VAR"]

tracer_provider = register(project_name="my-claude-app", batch=True)
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic(base_url=BASE, auth_token=TOKEN)

def ask(q: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": q}],
    )
    return r.content[0].text

# 1. A small test set: questions plus what a good answer should contain
df = pd.DataFrame({
    "input": [
        "What is the capital of France?",
        "What is 7 * 8?",
        "Why is the ocean salty?",
    ],
    "reference": [
        "Paris",
        "56",
        "Rivers carry dissolved minerals from rock weathering into the ocean, and evaporation leaves salt behind.",
    ],
})

# 2. Run your endpoint over it (each call is also traced in Phoenix)
df["output"] = df["input"].map(ask)

# 3. LLM-as-judge via the same gateway
judge = LLM(provider="anthropic", model=MODEL, base_url=BASE, auth_token=TOKEN)

correctness = create_classifier(
    name="correctness",
    llm=judge,
    prompt_template=(
        "Question: {input}\n"
        "Reference answer: {reference}\n"
        "Model answer: {output}\n\n"
        "Does the model answer agree with the reference? "
        "Respond 'correct' or 'incorrect'."
    ),
    choices={"correct": 1, "incorrect": 0},
)

results = evaluate_dataframe(dataframe=df, evaluators=[correctness])
print(results)
