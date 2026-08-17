"""Small dependency-light tests for the standalone benchmark components."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from ..dataset.datasets import load_samples  # pyright: ignore[reportMissingImports]
from ..dataset.tokenizer import BasicTokenizer  # pyright: ignore[reportMissingImports]
from ..lib.metrics import calculate_metrics  # pyright: ignore[reportMissingImports]
from ..lib.models import (  # pyright: ignore[reportMissingImports]
    RequestFuncOutput,
    SampleRequest,
)
from ..serve import benchmark, get_request  # pyright: ignore[reportMissingImports]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(size)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


class StandaloneBenchmarkTest(unittest.TestCase):
    def test_random_dataset_and_metrics(self) -> None:
        samples = load_samples("random", 2, 3, 2, BasicTokenizer())
        self.assertEqual([sample.prompt_len for sample in samples], [3, 3])
        output = RequestFuncOutput(
            success=True, prompt_len=3, output_tokens=2, latency=0.1
        )
        result = calculate_metrics([output], 0.1)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["total_output_tokens"], 2)

    def test_huggingface_dataset_is_normalized(self) -> None:
        fake_datasets = types.ModuleType("datasets")
        load_dataset = Mock(return_value=[{"text": "one two"}, {"prompt": "three"}])
        fake_datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            samples = load_samples(
                "org/example-dataset",
                3,
                0,
                5,
                BasicTokenizer(),
                dataset_config="default",
                dataset_split="validation",
            )

        load_dataset.assert_called_once_with(
            "org/example-dataset", name="default", split="validation"
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            {str(sample.prompt) for sample in samples}, {"one two", "three"}
        )
        self.assertTrue(all(sample.expected_output_len == 5 for sample in samples))

    def test_infinite_rate_emits_without_delay(self) -> None:
        async def collect() -> list[str | None]:
            samples = [SampleRequest("x", 1, request_id=str(i)) for i in range(3)]
            return [
                sample.request_id
                async for sample, _ in get_request(samples, float("inf"))
            ]

        self.assertEqual(asyncio.run(collect()), ["0", "1", "2"])

    def test_progress_tracks_completed_requests(self) -> None:
        progress = Mock()
        progress_factory = Mock(return_value=progress)
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = progress_factory  # type: ignore[attr-defined]
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            samples = [SampleRequest("say hi", 2, 2) for _ in range(2)]
            with patch.dict(sys.modules, {"tqdm": fake_tqdm}):
                result = asyncio.run(
                    benchmark(
                        samples,
                        backend="openai-chat",
                        base_url=f"http://127.0.0.1:{server.server_port}",
                        ready_check=False,
                        show_progress=True,
                    )
                )
            self.assertEqual(result["completed"], 2)
            progress_factory.assert_called_once_with(
                total=2, desc="Benchmark", unit="request"
            )
            self.assertEqual(progress.update.call_count, 2)
            progress.close.assert_called_once_with()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_chat_sse_benchmark(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            samples = [SampleRequest("say hi", 2, 2, request_id="case-0")]
            result = asyncio.run(
                benchmark(
                    samples,
                    backend="openai-chat",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model=None,
                    ready_check=True,
                    timeout_s=5,
                )
            )
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["total_output_tokens"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
