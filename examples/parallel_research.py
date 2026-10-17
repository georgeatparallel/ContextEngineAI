# /// script
# requires-python = ">=3.13"
# dependencies = ["mcp==1.29.1", "httpx>=0.28,<1"]
# ///
"""Collect cited web context without starting the research assistant's RAG stack."""

import argparse
import asyncio
import json
import sys
from datetime import timedelta
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "ContextEngineAI-Parallel-Example/0.1 "
    "(+https://github.com/patilrsohan/ContextEngineAI)"
)


def render_result(result) -> str:
    """Keep the service's excerpts and citation URLs together in the output."""
    text = "\n\n".join(block.text for block in result.content if block.type == "text")
    if not text and result.structuredContent is not None:
        text = json.dumps(result.structuredContent, indent=2, ensure_ascii=False)
    if result.isError:
        raise RuntimeError(text or "Parallel returned a tool error.")
    if not text.strip():
        raise RuntimeError("Parallel returned no text or structured research context.")
    return text


async def collect_context(tool: str, arguments: dict, timeout: float = 60) -> str:
    # Do not load .env, saved credentials or an Authorization header. This example
    # deliberately exercises the anonymous endpoint using the SDK's HTTP transport.
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=False
        ) as client:
            async with streamable_http_client(ENDPOINT, http_client=client) as streams:
                async with ClientSession(
                    streams[0], streams[1],
                    read_timeout_seconds=timedelta(seconds=timeout),
                    client_info=Implementation(name="contextengineai-parallel-example", version="0.1.0"),
                ) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if tool not in {item.name for item in tools.tools}:
                        raise RuntimeError(f"Parallel does not advertise {tool}.")
                    result = await session.call_tool(tool, arguments)
    return render_result(result)


def positive_seconds(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return number


def http_url(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        raise argparse.ArgumentTypeError("use an HTTP(S) URL without embedded credentials")
    return value


def error_message(error: Exception) -> str:
    # The SDK's concurrent transport can wrap network failures in task groups.
    if isinstance(error, ExceptionGroup):
        return "; ".join(error_message(item) for item in error.exceptions)
    return str(error) or type(error).__name__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=positive_seconds, default=60, help="total deadline in seconds")
    parser.add_argument("--session-id", default=None, help="reuse this ID for related search/fetch calls")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="find sources for a research objective")
    search.add_argument("objective")
    search.add_argument("--query", action="append", required=True, help="keyword query; repeat for related angles")
    fetch = commands.add_parser("fetch", help="read selected source URLs in more depth")
    fetch.add_argument("urls", nargs="+", type=http_url)
    fetch.add_argument("--objective", help="what to extract, at most 200 characters")
    fetch.add_argument("--query", action="append", help="queries that found these URLs")
    args = parser.parse_args(argv)
    session_id = args.session_id if args.session_id is not None else str(uuid4())
    if not session_id.strip() or len(session_id) > 100:
        parser.error("session ID must contain 1 to 100 characters")
    arguments = {"session_id": session_id}
    if args.command == "search":
        if not args.objective.strip() or any(not query.strip() for query in args.query):
            parser.error("objective and queries must not be blank")
        tool = "web_search"
        arguments.update(objective=args.objective, search_queries=args.query)
    else:
        if len(args.urls) > 20:
            parser.error("fetch accepts at most 20 URLs")
        if args.objective is not None and len(args.objective) > 200:
            parser.error("fetch objective must be at most 200 characters")
        tool = "web_fetch"
        arguments.update(urls=args.urls, full_content=False)
        if args.objective is not None:
            arguments["objective"] = args.objective
        if args.query is not None:
            arguments["search_queries"] = args.query
    print(f"Research session: {session_id}", file=sys.stderr)
    try:
        text = asyncio.run(collect_context(tool, arguments, args.timeout))
    except KeyboardInterrupt:
        print("Research cancelled.", file=sys.stderr)
        return 130
    except TimeoutError:
        print("Research timed out; try a narrower objective or a longer --timeout.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Research failed: {error_message(error)}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
