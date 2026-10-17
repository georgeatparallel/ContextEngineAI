"""Exercise the example through the MCP SDK and HTTP request boundary."""

import asyncio
import contextlib
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from mcp.types import CallToolResult, TextContent

import parallel_research as example

HTTP_CLIENT = httpx.AsyncClient


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.tool_result = {
            "content": [{"type": "text", "text": "# Retrieval study\nhttps://example.org/paper\nHybrid retrieval evidence."}],
            "isError": False,
        }
        self.call_started = asyncio.Event()
        self.block_call = False
        self.cancelled = False
        self.http_error = False

    async def respond(self, request):
        self.assertEqual(str(request.url), example.ENDPOINT)
        self.assertEqual(request.headers["User-Agent"], example.USER_AGENT)
        self.assertNotIn("Authorization", request.headers)
        body = json.loads(request.content)
        self.requests.append(body)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "research-fixture", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": [
                {"name": name, "inputSchema": {"type": "object"}}
                for name in ("web_search", "web_fetch")
            ]}
        elif method == "tools/call":
            self.call_started.set()
            if self.http_error:
                return httpx.Response(503, text="Service unavailable")
            if self.block_call:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            result = self.tool_result
        else:
            self.fail(f"Unexpected request: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def client(self, **kwargs):
        return HTTP_CLIENT(transport=httpx.MockTransport(self.respond), **kwargs)

    async def test_search_dispatch_and_cited_output(self):
        arguments = {
            "objective": "Find retrieval evaluations",
            "search_queries": ["hybrid retrieval evaluation research"],
            "session_id": "research-fixture-session",
        }
        with patch.object(example.httpx, "AsyncClient", self.client):
            text = await example.collect_context("web_search", arguments)
        self.assertIn("https://example.org/paper", text)
        self.assertIn("Hybrid retrieval evidence", text)
        methods = [item["method"] for item in self.requests]
        self.assertEqual(methods, ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        self.assertEqual(self.requests[-1]["params"], {"name": "web_search", "arguments": arguments})

    async def test_fetch_dispatch_and_cited_output(self):
        arguments = {"urls": ["https://example.org/paper"], "full_content": False, "session_id": "research-fixture-session"}
        with patch.object(example.httpx, "AsyncClient", self.client):
            text = await example.collect_context("web_fetch", arguments)
        self.assertIn("https://example.org/paper", text)
        self.assertEqual(self.requests[-1]["params"], {"name": "web_fetch", "arguments": arguments})

    async def test_tool_error_is_not_research_output(self):
        self.tool_result = {"content": [{"type": "text", "text": "Search quota exhausted"}], "isError": True}
        with patch.object(example.httpx, "AsyncClient", self.client):
            with self.assertRaisesRegex(RuntimeError, "Search quota exhausted"):
                await example.collect_context("web_search", {})
        self.assertTrue(self.call_started.is_set())

    async def test_http_failure_is_not_research_output(self):
        self.http_error = True
        with patch.object(example.httpx, "AsyncClient", self.client):
            with self.assertRaises(Exception):
                await example.collect_context("web_search", {}, timeout=1)
        self.assertTrue(self.call_started.is_set())

    async def test_deadline_cancels_in_flight_http(self):
        self.block_call = True
        with patch.object(example.httpx, "AsyncClient", self.client):
            with self.assertRaises(TimeoutError):
                await example.collect_context("web_search", {}, timeout=0.1)
        self.assertTrue(self.call_started.is_set())
        self.assertTrue(self.cancelled)

    async def test_cancellation_reaches_in_flight_http(self):
        self.block_call = True
        with patch.object(example.httpx, "AsyncClient", self.client):
            task = asyncio.create_task(example.collect_context("web_search", {}))
            await asyncio.wait_for(self.call_started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(self.cancelled)


class OutputTests(unittest.TestCase):
    def test_structured_content_and_empty_response(self):
        result = CallToolResult(content=[], structuredContent={"url": "https://example.org/paper"})
        self.assertIn("https://example.org/paper", example.render_result(result))
        with self.assertRaisesRegex(RuntimeError, "no text"):
            example.render_result(CallToolResult(content=[]))
        with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
            example.render_result(CallToolResult(content=[TextContent(type="text", text="quota exhausted")], isError=True))

    def test_search_cli_passes_queries_and_keeps_stdout_clean(self):
        run = AsyncMock(return_value="https://example.org/paper\nEvidence")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(example, "collect_context", run), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = example.main(["--session-id", "research-id", "search", "Find research", "--query", "retrieval research", "--query", "dense retrieval"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "https://example.org/paper\nEvidence\n")
        self.assertIn("research-id", stderr.getvalue())
        run.assert_awaited_once_with("web_search", {"session_id": "research-id", "objective": "Find research", "search_queries": ["retrieval research", "dense retrieval"]}, 60)

    def test_fetch_cli_and_session_reuse(self):
        run = AsyncMock(return_value="Evidence")
        with patch.object(example, "collect_context", run), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(example.main(["--session-id", "research-id", "fetch", "https://example.org/paper", "--objective", "Evaluation", "--query", "retrieval research"]), 0)
        run.assert_awaited_once_with("web_fetch", {"session_id": "research-id", "urls": ["https://example.org/paper"], "full_content": False, "objective": "Evaluation", "search_queries": ["retrieval research"]}, 60)

    def test_failures_exit_nonzero_without_success_output(self):
        for error in (RuntimeError("quota exhausted"), TimeoutError(), ExceptionGroup("transport", [httpx.ConnectError("Certificate verification failed")])):
            with self.subTest(error=type(error).__name__):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(example, "collect_context", AsyncMock(side_effect=error)), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = example.main(["search", "Find research", "--query", "retrieval research"])
                self.assertEqual(code, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertTrue(stderr.getvalue())
                if isinstance(error, ExceptionGroup):
                    self.assertIn("Certificate verification failed", stderr.getvalue())
                    self.assertNotIn("transport", stderr.getvalue())

    def test_invalid_inputs_do_not_connect(self):
        cases = [
            ["--timeout", "nan", "search", "topic", "--query", "research"],
            ["--session-id", "", "search", "topic", "--query", "research"],
            ["search", "", "--query", "research"],
            ["fetch", "file:///tmp/paper"],
            ["fetch", "https://user:secret@example.org"],
            ["fetch", "https://example.org", "--objective", "x" * 201],
            ["fetch"] + ["https://example.org"] * 21,
        ]
        for args in cases:
            with self.subTest(args=args), patch.object(example, "collect_context") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    example.main(args)
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
