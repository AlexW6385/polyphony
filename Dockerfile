FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# uv provides `uvx mcp-server-fetch` (the official Fetch MCP server) for senses.
RUN pip install uv

COPY pyproject.toml README.md ./
COPY polyphony ./polyphony
RUN pip install . && uv tool install mcp-server-fetch

COPY mind ./mind
COPY static ./static
COPY scripts ./scripts

EXPOSE 8000 8001
CMD ["python", "-m", "polyphony.main"]
