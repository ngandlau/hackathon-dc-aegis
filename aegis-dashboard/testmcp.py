from __future__ import annotations
import os, requests, anthropic, json

# ────────────────────────────────────────────────────────────
#  Firecrawl wrapper
# ────────────────────────────────────────────────────────────
def firecrawl_search(query: str, limit: int | None = 10) -> dict:
    """Call Firecrawl's /search endpoint and return a Claude-friendly dict."""
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"x-api-key": "fc-9146c3a11f7544239a5a07cbeadc7d2d"}
    params  = {"query": query, "numResults": limit or 10, "includeContent": False}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    # Strip the payload down to the essentials
    return {
        "results": [
            {
                "title": hit["title"],
                "url":   hit["url"],
                "snippet": hit.get("snippet", "")
            }
            for hit in r.json().get("results", [])
        ]
    }

# ────────────────────────────────────────────────────────────
#  Tool manifest (must match exactly on every request)
# ────────────────────────────────────────────────────────────
firecrawl_tool = {
    "name": "firecrawl_search",
    "description": "Searches the public web via Firecrawl.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",  "description": "Search query"},
            "limit": {"type": "integer", "description": "Number of results"}
        },
        "required": ["query"]
    }
}

# ────────────────────────────────────────────────────────────
#  Conversation loop
# ────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

messages: list[dict] = [
    {
        "role": "user",
        "content": "Use the firecrawl_search tool to find the latest public health trends in the US."
    }
]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[firecrawl_tool],
        messages=messages
    )

    # When Claude is done, print the answer and exit
    if response.stop_reason in ("end", "end_turn"):
        print("\nAssistant:\n")
        for block in response.content:
            # Text blocks arrive as TextBlock objects
            if block.type == "text":
                print(block.text)
        break

    # Otherwise Claude is requesting one or more tool calls
    if response.stop_reason != "tool_use":
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

    # Iterate over every ToolUseBlock in the response
    for block in response.content:
        if block.type != "tool_use":
            continue
        if block.name != "firecrawl_search":
            raise RuntimeError(f"Unknown tool: {block.name}")

        print(f"\n→ Claude called {block.name} with {json.dumps(block.input)}")

        # Run the real tool (or stub, if you prefer)
        try:
            tool_output = firecrawl_search(**block.input)
        except Exception as exc:
            tool_output = {"error": str(exc)}

        # Push the assistant's tool-use turn and our tool_result back into history
        messages.extend([
            {
                "role": "assistant",
                "content": response.content  # the exact blocks we just got
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_output)  # must be a string
                    }
                ]
            }
        ])

# ────────────────────────────────────────────────────────────
#  End of script
# ────────────────────────────────────────────────────────────
