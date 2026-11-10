# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Framework x Accelerator RuntimeBootstrap plugins."""

from __future__ import annotations

import builtins
import importlib
import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.distributed.runtime import bootstraps
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.distributed.runtime.bootstraps import bootstrap_api
from tunix.dist_rl.distributed.runtime.bootstraps import jax_gpu
from tunix.dist_rl.distributed.runtime.bootstraps import jax_tpu
from tunix.dist_rl.distributed.runtime.bootstraps import torch_tpu
from tunix.dist_rl.weight_sync import raiden_preload
from tunix.dist_rl.weight_sync import weight_sync


class BootstrapsTest(parameterized.TestCase):

  @parameterized.parameters(
      ("jax:tpu", jax_tpu.JaxTpuBootstrap),
      ("jax:gpu", jax_gpu.JaxGpuBootstrap),
      ("torch:tpu", torch_tpu.TorchTpuBootstrap),
  )
  def test_resolves_registered_bootstraps_without_dynamic_import(
      self, spec: str, expected_cls: type[bootstraps.RuntimeBootstrap]
  ):
    target = execution_target.ExecutionTarget.parse(spec)
    with mock.patch.object(
        builtins, "__import__", wraps=builtins.__import__
    ) as mock_import:
      plugin = bootstraps.get_runtime_bootstrap(target)
    self.assertIsInstance(plugin, expected_cls)
    self.assertEqual(plugin.framework, target.framework)
    self.assertEqual(plugin.accelerator, target.accelerator)
    mock_import.assert_not_called()

  def test_lazy_import_populates_missing_builtin_bootstrap(self):
    saved = bootstrap_api.BOOTSTRAP_REGISTRY._registry.pop("jax:gpu")
    orig_import = builtins.__import__
    try:

      def _fake_import(name: str, *args, **kwargs):
        if name == "tunix.dist_rl.distributed.runtime.bootstraps.jax_gpu":
          bootstrap_api.BOOTSTRAP_REGISTRY._registry["jax:gpu"] = saved
          return jax_gpu
        return orig_import(name, *args, **kwargs)

      # Callers such as `rollout.py` and `run_rollout_node.py` have unit tests
      # that patch `importlib.import_module`; `get_runtime_bootstrap` must
      # still lazily load builtin bootstrap modules via `__import__`.
      with (
          mock.patch.object(importlib, "import_module"),
          mock.patch.object(
              builtins, "__import__", side_effect=_fake_import
          ) as mock_import,
      ):
        plugin = bootstrap_api.get_runtime_bootstrap(
            execution_target.ExecutionTarget.parse("jax:gpu")
        )
      self.assertIsInstance(plugin, jax_gpu.JaxGpuBootstrap)
      mock_import.assert_called_once_with(
          "tunix.dist_rl.distributed.runtime.bootstraps.jax_gpu",
          fromlist=["*"],
      )
    finally:
      bootstrap_api.BOOTSTRAP_REGISTRY._registry["jax:gpu"] = saved

  @parameterized.parameters(
      (weight_sync.WeightSyncMode.NONE, "", False),
      (weight_sync.WeightSyncMode.FS, "0", False),
      (weight_sync.WeightSyncMode.IN_MEMORY, "", False),
      (weight_sync.WeightSyncMode.RAIDEN, "", True),
      (weight_sync.WeightSyncMode.NONE, "true", True),
  )
  def test_jax_tpu_bootstrap_raiden_and_jax_init(
      self,
      mode: weight_sync.WeightSyncMode,
      use_raiden_env: str,
      expect_raiden: bool,
  ):
    ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
    env = {"USE_RAIDEN": use_raiden_env} if use_raiden_env else {}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(raiden_preload, "import_raiden") as mock_raiden,
    ):
      jax_tpu.JaxTpuBootstrap().initialize(ctx, weight_sync_mode=mode)
    self.assertEqual(mock_raiden.call_count, 1 if expect_raiden else 0)
    ctx.jax.initialize.assert_called_once_with()

  def test_jax_tpu_bootstrap_skips_jax_init_when_disabled(self):
    ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
    with mock.patch.dict(os.environ, {}, clear=True):
      jax_tpu.JaxTpuBootstrap().initialize(
          ctx,
          weight_sync_mode=weight_sync.WeightSyncMode.NONE,
          initialize_runtime=False,
      )
    ctx.jax.initialize.assert_not_called()

  def test_jax_gpu_bootstrap_sets_cuda_defaults_without_raiden(self):
    ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(raiden_preload, "import_raiden") as mock_raiden,
    ):
      jax_gpu.JaxGpuBootstrap().initialize(
          ctx, weight_sync_mode=weight_sync.WeightSyncMode.NCCL
      )
      self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda,cpu")
      self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
      self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85")
    mock_raiden.assert_not_called()
    ctx.jax.initialize.assert_called_once_with()

  def test_torch_tpu_bootstrap_sets_pjrt_device_and_imports_tpu_backend(self):
    ctx = mock.create_autospec(context_lib.ProcessContext, instance=True)
    with (
        mock.patch.dict(
            os.environ, {"TORCH_TPU_MODULE": "torch_tpu"}, clear=True
        ),
        mock.patch.object(importlib, "import_module") as mock_import,
    ):
      torch_tpu.TorchTpuBootstrap().initialize(ctx)
      self.assertEqual(os.environ["PJRT_DEVICE"], "TPU")
    self.assertEqual(
        mock_import.call_args_list,
        [mock.call("torch"), mock.call("torch_tpu")],
    )
    ctx.jax.initialize.assert_not_called()

    # Empty TORCH_TPU_MODULE defaults to torchtpu, and
    # initialize_runtime=False still sets PJRT_DEVICE=TPU without importing
    # torch.
    with (
        mock.patch.dict(os.environ, {"TORCH_TPU_MODULE": ""}, clear=True),
        mock.patch.object(importlib, "import_module") as mock_import_deferred,
    ):
      torch_tpu.TorchTpuBootstrap().initialize(ctx, initialize_runtime=False)
      self.assertEqual(os.environ["PJRT_DEVICE"], "TPU")
    mock_import_deferred.assert_not_called()

    with (
        mock.patch.dict(os.environ, {"TORCH_TPU_MODULE": "  "}, clear=True),
        mock.patch.object(importlib, "import_module") as mock_import_empty,
    ):
      torch_tpu.TorchTpuBootstrap().initialize(ctx)
    self.assertEqual(
        mock_import_empty.call_args_list,
        [mock.call("torch"), mock.call("torchtpu")],
    )

  def test_import_bootstraps_has_zero_ml_framework_and_plugin_imports(self):
    forbidden = ("jax", "flax", "optax", "torch", "tpu_raiden")
    with mock.patch.dict(sys.modules):
      for name in list(sys.modules):
        if "tunix" in name.split("."):
          sys.modules.pop(name, None)
        elif any(pkg in name.split(".") for pkg in forbidden):
          sys.modules[name] = None
      for pkg in forbidden:
        sys.modules[pkg] = None
      mod = importlib.import_module(
          "tunix.dist_rl.distributed.runtime.bootstraps"
      )
      self.assertFalse(mod.BOOTSTRAP_REGISTRY.contains("jax:tpu"))
      self.assertFalse(mod.BOOTSTRAP_REGISTRY.contains("torch:tpu"))
      plugin = mod.get_runtime_bootstrap(
          execution_target.ExecutionTarget.parse("jax:gpu")
      )
      self.assertTrue(mod.BOOTSTRAP_REGISTRY.contains("jax:gpu"))
      self.assertFalse(mod.BOOTSTRAP_REGISTRY.contains("torch:tpu"))
      self.assertEqual(plugin.accelerator, execution_target.Accelerator.GPU)


if __name__ == "__main__":
  absltest.main()
