"""A GDN tree verify that keeps replay records in place of per-step snapshots - CPU only.

Under ``--enable-linear-replayssm-spec`` a tree draft (top-k > 1) cannot use the
chain ring. A speculative algorithm whose verify extension keeps replay records
(``linear_attn_verify_record_width``) gets them from the pool instead of the
per-step snapshots and the ring: the extension sees them in its state layout,
per layer as ``intermediate_states``, and commits accepted states from them.

Backends and the configurator are built with ``__new__`` and only the
attributes each method reads.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.linear import gdn_backend as gdn_mod
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.verify_extension import (
    LinearAttnStateLayout,
    LinearAttnVerifyExtension,
)
from sglang.srt.mem_cache import kv_cache_configurator as configurator_mod
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

HV, V, K = 2, 8, 4
LAYERS = [0, 1]
WIDTH = 4  # verify steps
RECORD = V + K + 4  # the record width a plugin might ask for


def _gdn_params():
    shape = Mamba2StateShape(
        conv=[(16, 3)],
        temporal=(HV, V, K),
        intermediate_size=0,
        conv_dim=16,
        ssm_state_size=K,
        num_heads=HV,
        head_dim=V,
        state_size=K,
        conv_kernel=4,
        num_k_heads_per_tp=1,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=LAYERS)


def _pool(**kwargs):
    return MambaPool(
        size=3,
        spec_state_size=2,
        cache_params=_gdn_params(),
        mamba_layer_ids=LAYERS,
        device="cpu",
        speculative_num_draft_tokens=WIDTH,
        speculative_eagle_topk=4,
        **kwargs,
    )


class TestPool(CustomTestCase):
    def test_records_replace_the_snapshots_and_the_ring(self):
        pool = _pool(linear_verify_record_width=RECORD)
        cache = pool.mamba_cache
        self.assertIsNone(cache.intermediate_ssm)
        self.assertIsNone(cache.replayssm_d)
        self.assertIsNone(pool.replayssm_cache_base)
        self.assertEqual(cache.verify_records.dtype, torch.float32)
        # [layers, spec_state_size + 1 rows, steps, value heads, width]
        self.assertEqual(tuple(cache.verify_records.shape), (2, 3, WIDTH, HV, RECORD))
        layer = cache.at_layer_idx(1)
        self.assertEqual(layer.verify_records.data_ptr(), cache.verify_records[1].data_ptr())
        # the conv windows still roll the conv state back
        self.assertEqual(len(cache.intermediate_conv_window), 1)
        self.assertIn("verify_records", MambaPool._NON_TRANSFER_STATE_FIELDS)

    def test_default_keeps_the_snapshots(self):
        cache = _pool().mamba_cache
        self.assertIsNone(cache.verify_records)
        self.assertEqual(tuple(cache.intermediate_ssm.shape), (2, 3, WIDTH, HV, V, K))


class _Records(LinearAttnVerifyExtension):
    def __init__(self, handle=True):
        self.handle = handle
        self.states = None
        self.attn_out = torch.full((1, WIDTH, HV, V), 7.0)

    def conv_update(self, layer, mixed_qkv, **kwargs):
        return mixed_qkv

    def recurrent_update(self, layer, *, intermediate_states, **kwargs):
        self.states = intermediate_states
        return self.attn_out if self.handle else None


def _verify_backend(extension):
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.verify_extension = extension
    backend._verify_extension_layout = object()  # begin_forward saw a tree verify
    backend._use_strided_target_verify_qkv = False
    backend.mis_metadata = None
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, WIDTH], dtype=torch.int32),
        mamba_cache_indices=torch.zeros(1, dtype=torch.int32),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    layer_cache = MambaPool.SpeculativeState(
        conv=[torch.zeros(2, 32, 3)],
        temporal=torch.zeros(2, HV, V, K),
        intermediate_ssm=None,
        intermediate_conv_window=[torch.zeros(2, WIDTH, 32, 3)],
        verify_records=torch.zeros(2, WIDTH, HV, RECORD),
    )
    backend.req_to_token_pool = SimpleNamespace(
        size=4,
        mamba2_layer_cache=lambda layer_id: layer_cache,
        mamba_pool=SimpleNamespace(),
    )
    backend.verify_intermediate_state_indices = torch.arange(4, dtype=torch.int32)
    backend.kernel_dispatcher = mock.MagicMock()
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=K,
        k_dim=K,
        v_dim=HV * V,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=HV,
        head_q_dim=K,
        head_k_dim=K,
        head_v_dim=V,
        conv_weights=None,
        bias=None,
        activation="silu",
        A_log=None,
        dt_bias=None,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=WIDTH),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
    )
    mixed_qkv = torch.randn(WIDTH, 2 * K + HV * V)
    return backend, layer, forward_batch, mixed_qkv, layer_cache


class TestVerify(CustomTestCase):
    def setUp(self):
        # CPU tensors: split q/k/v with torch instead of the Triton kernel.
        patcher = mock.patch.object(gdn_mod, "is_cuda", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, extension):
        backend, layer, fb, mixed_qkv, layer_cache = _verify_backend(extension)
        out = backend.forward_extend(
            layer, fb, mixed_qkv, a=torch.zeros(WIDTH, HV), b=torch.zeros(WIDTH, HV)
        )
        return backend, out, layer_cache

    def test_the_extension_gets_the_layer_records(self):
        extension = _Records()
        backend, out, layer_cache = self._run(extension)
        self.assertIs(extension.states, layer_cache.verify_records)
        self.assertIs(out, extension.attn_out)
        backend.kernel_dispatcher.target_verify.assert_not_called()

    def test_a_verify_the_extension_leaves_runs_without_snapshots(self):
        # warmup and autotune dummy runs carry no tree links; they never commit
        sentinel = torch.zeros(1)
        with mock.patch(
            "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent."
            "fused_sigmoid_gating_delta_rule_update",
            return_value=sentinel,
        ) as kernel:
            backend, out, _ = self._run(_Records(handle=False))
        self.assertIs(out, sentinel)
        self.assertIsNone(kernel.call_args.kwargs["intermediate_states_buffer"])
        self.assertTrue(kernel.call_args.kwargs["disable_state_update"])
        backend.kernel_dispatcher.target_verify.assert_not_called()

    def test_tree_layout_accepts_records_but_not_the_ring(self):
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        backend.verify_extension = mock.MagicMock()
        links = torch.zeros(1, WIDTH, dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            retrieve_next_token=links,
            retrieve_next_sibling=links,
            retrieve_parent_token=links,
            mamba_cache_indices=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.tensor([0, WIDTH], dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=WIDTH),
        )
        records = SimpleNamespace(intermediate_ssm=None, verify_records=torch.zeros(1))
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=records)
        )
        backend._begin_verify_extension(forward_batch, in_graph=False)
        self.assertIsNotNone(backend._verify_extension_layout)
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=records, replayssm_cache_base=1)
        )
        with self.assertRaises(RuntimeError):
            backend._begin_verify_extension(forward_batch, in_graph=False)


class TestCommit(CustomTestCase):
    def test_records_without_a_commit_refuse_the_snapshot_scatter(self):
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid_mod

        slots = torch.tensor([1], dtype=torch.int32)
        cache = SimpleNamespace(temporal=torch.zeros(2, 2, HV, V, K), verify_records=torch.zeros(1))
        for handled in (True, False):
            extension = SimpleNamespace(commit_state=mock.Mock(return_value=handled))
            linear = SimpleNamespace(
                forward_metadata=SimpleNamespace(
                    mamba_cache_indices=slots,
                    retrieve_parent_token=torch.zeros(1, WIDTH, dtype=torch.int32),
                ),
                req_to_token_pool=SimpleNamespace(
                    get_speculative_mamba2_params_all_layers=lambda: cache,
                    mamba_pool=SimpleNamespace(replayssm_is_kda=False),
                ),
                verify_extension=extension,
                verify_intermediate_state_indices=torch.zeros(1, dtype=torch.int32),
                accept_lens_pool=None,
                _translate_mamba_indices=lambda value: value,
            )
            backend = hybrid_mod.HybridLinearAttnBackend.__new__(
                hybrid_mod.HybridLinearAttnBackend
            )
            backend.linear_attn_backend = linear
            backend._update_ple_state_after_mtp_verify = mock.Mock()
            steps = torch.tensor([3], dtype=torch.int32)
            with (
                mock.patch.object(hybrid_mod, "pp_spec_stable_rows_enabled", return_value=False),
                mock.patch.object(hybrid_mod, "scatter_mamba_states_after_mtp_verify") as scatter,
            ):
                if handled:
                    backend.update_mamba_state_after_mtp_verify(steps, None, None, None)
                    self.assertTrue(scatter.call_args.kwargs["skip_ssm"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "replay records"):
                        backend.update_mamba_state_after_mtp_verify(steps, None, None, None)
                    scatter.assert_not_called()


class _Plugin(CustomSpecAlgo):
    def __init__(self, width):
        super().__init__(name="RECORDS_TEST", factory=None)
        self.width = width
        self.asked = []

    def linear_attn_verify_record_width(self, server_args, *, head_v_dim, head_k_dim):
        self.asked.append((head_v_dim, head_k_dim))
        return self.width


class TestConfigurator(CustomTestCase):
    def _width(self, *, flag=True, topk=4, algorithm=None, gdn=True):
        configurator = configurator_mod.KVCacheConfigurator.__new__(
            configurator_mod.KVCacheConfigurator
        )
        params = _gdn_params()
        configurator.hybrid_gdn_config = object() if gdn else None
        configurator.mambaish_config = SimpleNamespace(mamba2_cache_params=params)
        configurator.spec_algorithm = algorithm or _Plugin(RECORD)
        configurator.server_args = object()
        exec_ = SimpleNamespace(mamba=SimpleNamespace(enable_linear_replayssm_spec=flag))
        spec = SimpleNamespace(speculative_eagle_topk=topk)
        with (
            mock.patch.object(configurator_mod, "get_exec", return_value=exec_),
            mock.patch.object(configurator_mod, "get_spec", return_value=spec),
        ):
            return configurator._linear_verify_record_width()

    def test_a_tree_under_the_flag_asks_the_algorithm(self):
        plugin = _Plugin(RECORD)
        self.assertEqual(self._width(algorithm=plugin), RECORD)
        self.assertEqual(plugin.asked, [(V, K)])

    def test_chains_and_unflagged_trees_keep_their_paths(self):
        plugin = _Plugin(RECORD)
        self.assertEqual(self._width(flag=False, algorithm=plugin), 0)
        self.assertEqual(self._width(topk=1, algorithm=plugin), 0)
        self.assertEqual(self._width(topk=None, algorithm=plugin), 0)
        self.assertEqual(plugin.asked, [])

    def test_a_tree_without_records_is_refused(self):
        with self.assertRaisesRegex(ValueError, "replay records"):
            self._width(algorithm=_Plugin(0))
        with self.assertRaisesRegex(ValueError, "replay records"):
            self._width(algorithm=SpeculativeAlgorithm.EAGLE)
        with self.assertRaisesRegex(ValueError, "GDN target"):
            self._width(gdn=False)


class TestDefaults(CustomTestCase):
    def test_builtins_and_plugins_keep_snapshots_by_default(self):
        plugin = CustomSpecAlgo(name="DEFAULT_TEST", factory=None)
        for algorithm in (SpeculativeAlgorithm.EAGLE, plugin):
            self.assertEqual(
                algorithm.linear_attn_verify_record_width(
                    None, head_v_dim=V, head_k_dim=K
                ),
                0,
            )
        self.assertIsNone(
            LinearAttnStateLayout(
                conv_state_len=3,
                conv_dtype=torch.bfloat16,
                ssm_dtype=torch.float32,
                num_v_heads=HV,
                has_state_snapshots=True,
                device=torch.device("cpu"),
            ).verify_records
        )


if __name__ == "__main__":
    unittest.main()
