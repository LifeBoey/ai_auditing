import os
from anthropic import Anthropic
from phoenix.otel import register
from openinference.instrumentation.anthropic import AnthropicInstrumentor

tracer_provider = register(project_name="my-claude-app")
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic(
    base_url=os.environ["YOUR_BASE_URL_VAR"],
    auth_token=os.environ["YOUR_AUTH_TOKEN_VAR"],   # sent as a Bearer token
)

resp = client.messages.create(
    model=os.environ["YOUR_MODEL_VAR"],
    max_tokens=500,
    messages=[{"role": "user", "content": "Why is the ocean salty?"}],
)
print(resp.content[0].text)
