"""
examples/vercel_server.py — OpenAI-compatible HTTP server for Vercel AI SDK,
LlamaIndex, curl, and any OpenAI-compatible client.

Starts a FastAPI server at http://localhost:8000 that wraps BAARRouter as a
/v1/chat/completions endpoint. Any client that speaks the OpenAI API works
without code changes.

Install:
    pip install baar-core[vercel]                    # Python server
    pip install llama-index-llms-openai              # LlamaIndex (optional)
    npm install ai @ai-sdk/openai                    # Vercel AI SDK (optional)

Run the server:
    python examples/vercel_server.py

─────────────────────────────────────────────────────────────────────────────
Client examples (once server is running)
─────────────────────────────────────────────────────────────────────────────

curl:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"baar","messages":[{"role":"user","content":"Hello"}]}'

Vercel AI SDK (TypeScript/JavaScript):
    import { createOpenAI } from '@ai-sdk/openai';
    import { streamText } from 'ai';

    const baar = createOpenAI({
        baseURL: 'http://localhost:8000/v1',
        apiKey: 'your-secret',        // matches api_key below
    });

    const { textStream } = streamText({
        model: baar('baar'),
        messages: [{ role: 'user', content: 'Explain the GIL in Python' }],
    });

    for await (const chunk of textStream) {
        process.stdout.write(chunk);
    }

LlamaIndex (Python):
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.core import Settings

    llm = LlamaOpenAI(
        model='baar',
        api_base='http://localhost:8000/v1',
        api_key='your-secret',
    )
    Settings.llm = llm

    response = llm.complete('Summarise the CAP theorem')
    print(response)

OpenAI Python SDK:
    from openai import OpenAI

    client = OpenAI(
        base_url='http://localhost:8000/v1',
        api_key='your-secret',
    )

    response = client.chat.completions.create(
        model='baar',
        messages=[{'role': 'user', 'content': 'What is recursion?'}],
    )
    print(response.choices[0].message.content)
─────────────────────────────────────────────────────────────────────────────

Budget errors surface as standard HTTP codes:
    402  Budget exhausted — no more calls allowed
    422  Task rejected   — value gate blocked the request
    401  Unauthorized    — wrong or missing api_key
"""

from baar import BAARRouter
from baar.integrations.vercel import create_app

# ── Configure router ──────────────────────────────────────────────────────────

router = BAARRouter(
    budget=1.00,                     # hard cap: $1.00 total
    small_model="gpt-4o-mini",       # cheap tier
    big_model="gpt-4o",              # capable tier
    complexity_threshold=0.80,       # 0–1: higher = more traffic to cheap model
)

# ── Create FastAPI app ────────────────────────────────────────────────────────

app = create_app(
    router,
    api_key="your-secret",           # set None to disable auth (localhost only)
)

# ── Start server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn not installed. Run: pip install baar-core[vercel]")

    print("Baar OpenAI-compatible server")
    print(f"  Budget:      ${router.budget:.2f}")
    print(f"  Small model: {router.small_model}")
    print(f"  Big model:   {router.big_model}")
    print(f"  Endpoint:    http://localhost:8000/v1/chat/completions")
    print(f"  Models:      http://localhost:8000/v1/models")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
