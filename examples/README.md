# Research sources with Parallel Search MCP

Use this standalone example to find current web sources to compare with your
local research documents. It prints the service's excerpts and source URLs so you
can inspect the evidence before using it in research notes. It does not synthesize
an answer or feed results into the Streamlit/CrewAI workflow.

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) offers
free anonymous search in Fast mode and page fetching, with no account or API key
needed for light use. Anonymous usage has rate limits. The script uses the MCP
Python SDK over Streamable HTTP and sends no authentication header.

## Run from the repository root

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv run --script examples/parallel_research.py search \
  "Find recent evaluations comparing hybrid retrieval with dense retrieval for RAG" \
  --query "hybrid dense retrieval RAG benchmarks" \
  --query "hybrid retrieval evaluation research"
```

`uv run --script` installs only the dependencies declared in the script into an
isolated environment and selects Python 3.13 or later. The application's RAG
dependencies, database and `.env` are not needed for this example.

Search excerpts are often enough to assess a source. When you need more detail,
choose a source URL and fetch its relevant passages:

```bash
uv run --script examples/parallel_research.py fetch https://arxiv.org/abs/2004.04906 \
  --objective "Find the retrieval method and evaluation datasets"
```

To fetch a URL found by your search, substitute that URL and pass the same
`--query` values. Reuse the `Research session` ID printed to stderr across related
calls by placing `--session-id YOUR_SESSION_ID` before `search` or `fetch`.
The script creates a new session ID when you omit it; use a new ID for a new
research task. Fetch accepts up to 20 URLs and an optional objective of at most
200 characters. It requests focused excerpts rather than full pages.

Output goes to stdout, so append `> research-context.md` to save it. Review the
source URLs and excerpts before drawing conclusions; returned page text is
external content. Network, server and quota errors exit with code 1 and report
the error to stderr. Calls have a 60 second total deadline; put `--timeout 120`
before the subcommand to change it. Ctrl+C cancels the request.

The existing Firecrawl web search, application setup and provider choices are
unchanged. The other application services still require their existing keys when
you run the full assistant.

## Check the example

The tests use the real SDK with a simulated HTTP server response, without network
access or API keys:

```bash
uv run --no-project --python 3.13 --with mcp==1.29.1 --with 'httpx>=0.28,<1' \
  python -m unittest discover -s examples -p 'test_parallel_research.py'
```
