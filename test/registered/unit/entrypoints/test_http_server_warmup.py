import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class TestSamplingServerWarmup(CustomTestCase):
    def _run_warmup(self, *, generation=True, vlm=False, fail_sampled=False):
        from sglang.srt.entrypoints import http_server
        from sglang.srt.managers.tokenizer_manager import ServerStatus
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(
            dp_size=1,
            disaggregation_mode="null",
            skip_tokenizer_init=False,
            debug_tensor_dump_input_file=None,
            api_key=None,
            language_only=False,
            language_model_only=False,
        )
        override.install()
        self.addCleanup(override.restore)
        manager = SimpleNamespace(server_status=None, served_model_name="test")
        calls = []

        def post(url, **kwargs):
            self.assertNotEqual(manager.server_status, ServerStatus.Up)
            payload = copy.deepcopy(kwargs["json"])
            calls.append((url, payload))
            sampled = payload.get("sampling_params", payload).get("temperature") == 1.0
            return SimpleNamespace(
                status_code=500 if sampled and fail_sampled else 200, text="response"
            )

        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"is_generation": generation, "has_image_understanding": vlm},
        )
        with (
            patch.object(http_server.time, "sleep"),
            patch.object(http_server.requests, "get", return_value=response),
            patch.object(http_server.requests, "post", side_effect=post),
            patch.object(http_server, "ssl_verify_of", return_value=False),
            patch.object(http_server, "is_mps", return_value=False),
            patch.object(http_server, "kill_process_tree"),
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            ),
        ):
            success = http_server._execute_server_warmup(
                SimpleNamespace(url=lambda: "http://localhost:30000")
            )
        return success, calls, manager.server_status

    def test_generation_warms_greedy_and_sampled_paths_before_ready(self):
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        success, calls, status = self._run_warmup()
        self.assertTrue(success)
        self.assertEqual(
            [payload["sampling_params"]["temperature"] for _, payload in calls],
            [0, 1.0],
        )
        self.assertEqual(calls[1][1]["sampling_params"]["top_p"], 0.9)
        self.assertEqual(status, ServerStatus.Up)
        self.assertTrue(all(url.endswith("/generate") for url, _ in calls))

    def test_vlm_warms_the_chat_sampling_path(self):
        success, calls, _ = self._run_warmup(vlm=True)
        self.assertTrue(success)
        self.assertEqual([payload["temperature"] for _, payload in calls], [0, 1.0])
        self.assertTrue(all(url.endswith("/v1/chat/completions") for url, _ in calls))

    def test_embeddings_do_not_run_a_sampling_warmup(self):
        success, calls, _ = self._run_warmup(generation=False)
        self.assertTrue(success)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0].endswith("/encode"))

    def test_sampled_warmup_failure_does_not_publish_readiness(self):
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        success, calls, status = self._run_warmup(fail_sampled=True)
        self.assertFalse(success)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(status, ServerStatus.Up)


if __name__ == "__main__":
    unittest.main()
