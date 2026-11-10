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

"""Tests for role-scoped ExecutionTarget and ClusterTargets contracts."""

import importlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import datatypes as dist_datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.weight_sync import weight_sync


class ExecutionTargetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="jax_tpu_default_engine",
          spec="jax:tpu",
          default_engine="peft_trainer_v2",
          expected_framework=execution_target.Framework.JAX,
          expected_accelerator=execution_target.Accelerator.TPU,
          expected_engine="peft_trainer_v2",
          expected_formatted="jax:tpu:peft_trainer_v2",
      ),
      dict(
          testcase_name="jax_gpu_no_engine",
          spec="jax:gpu",
          default_engine="",
          expected_framework=execution_target.Framework.JAX,
          expected_accelerator=execution_target.Accelerator.GPU,
          expected_engine="",
          expected_formatted="jax:gpu",
      ),
      dict(
          testcase_name="jax_gpu_explicit_engine",
          spec="jax:gpu:vllm",
          default_engine="inprocess_vllm",
          expected_framework=execution_target.Framework.JAX,
          expected_accelerator=execution_target.Accelerator.GPU,
          expected_engine="vllm",
          expected_formatted="jax:gpu:vllm",
      ),
      dict(
          testcase_name="torch_tpu_explicit_engine",
          spec="torch:tpu:torchtitan",
          default_engine="",
          expected_framework=execution_target.Framework.TORCH,
          expected_accelerator=execution_target.Accelerator.TPU,
          expected_engine="torchtitan",
          expected_formatted="torch:tpu:torchtitan",
      ),
  )
  def test_parse_and_format_roundtrip(
      self,
      spec: str,
      default_engine: str,
      expected_framework: execution_target.Framework,
      expected_accelerator: execution_target.Accelerator,
      expected_engine: str,
      expected_formatted: str,
  ):
    target = execution_target.ExecutionTarget.parse(
        spec, default_engine=default_engine
    )
    self.assertEqual(target.framework, expected_framework)
    self.assertEqual(target.accelerator, expected_accelerator)
    self.assertEqual(target.engine, expected_engine)
    self.assertEqual(target.format(), expected_formatted)

  @parameterized.parameters(
      "",
      " jax:tpu",
      "jax",
      "jax:tpu:",
      ":tpu",
      "jax::vllm",
      "jax:tpu:vllm:extra",
      "mlx:gpu",
      "jax:npu",
  )
  def test_parse_rejects_malformed_specs(self, bad_spec: str):
    with self.assertRaises(ValueError):
      execution_target.ExecutionTarget.parse(bad_spec)

  def test_parse_base_shorthand_rejects_engine_segment(self):
    with self.assertRaisesRegex(
        ValueError, "expected '<framework>:<accelerator>'"
    ):
      execution_target.ExecutionTarget.parse(
          "jax:gpu:vllm", allow_engine=False
      )

  def test_engine_rejects_colon_and_whitespace(self):
    with self.assertRaises(ValueError):
      execution_target.ExecutionTarget(
          framework=execution_target.Framework.JAX,
          accelerator=execution_target.Accelerator.TPU,
          engine="bad:engine",
      )
    with self.assertRaises(ValueError):
      execution_target.ExecutionTarget(
          framework=execution_target.Framework.JAX,
          accelerator=execution_target.Accelerator.TPU,
          engine=" padded ",
      )

  def test_cluster_targets_accepts_disaggregated_cross_accelerator(self):
    targets = execution_target.ClusterTargets(
        trainer=execution_target.ExecutionTarget.parse("torch:tpu:torchtitan"),
        rollout=execution_target.ExecutionTarget.parse("jax:gpu:vllm"),
        inference=execution_target.ExecutionTarget.parse("jax:tpu:reference"),
        placement=execution_target.Placement.DISAGGREGATED,
        weight_sync_mode=weight_sync.WeightSyncMode.FS,
    )
    self.assertEqual(
        targets.trainer.accelerator, execution_target.Accelerator.TPU
    )
    self.assertEqual(
        targets.rollout.accelerator, execution_target.Accelerator.GPU
    )

  def test_cluster_targets_rejects_colocated_mismatched_accelerators(self):
    with self.assertRaisesRegex(ValueError, "Colocated placement"):
      execution_target.ClusterTargets(
          trainer=execution_target.ExecutionTarget.parse("jax:tpu"),
          rollout=execution_target.ExecutionTarget.parse("jax:gpu"),
          inference=execution_target.ExecutionTarget.parse("jax:tpu"),
          placement=execution_target.Placement.COLOCATED,
      )
    with self.assertRaisesRegex(ValueError, "Colocated placement"):
      execution_target.ClusterTargets(
          trainer=execution_target.ExecutionTarget.parse("jax:gpu"),
          rollout=execution_target.ExecutionTarget.parse("jax:gpu"),
          inference=execution_target.ExecutionTarget.parse("torch:tpu"),
          placement=execution_target.Placement.COLOCATED,
      )

  def test_cluster_targets_accepts_in_memory_for_colocated_and_disaggregated(
      self,
  ):
    for placement in (
        execution_target.Placement.COLOCATED,
        execution_target.Placement.DISAGGREGATED,
    ):
      targets = execution_target.ClusterTargets(
          trainer=execution_target.ExecutionTarget.parse("jax:gpu"),
          rollout=execution_target.ExecutionTarget.parse("jax:gpu"),
          inference=execution_target.ExecutionTarget.parse("jax:gpu"),
          placement=placement,
          weight_sync_mode=weight_sync.WeightSyncMode.IN_MEMORY,
      )
      self.assertEqual(targets.placement, placement)
      self.assertEqual(
          targets.weight_sync_mode, weight_sync.WeightSyncMode.IN_MEMORY
      )

  def test_cluster_targets_validates_raiden_requires_tpu_endpoints(self):
    valid = execution_target.ClusterTargets(
        trainer=execution_target.ExecutionTarget.parse("jax:tpu"),
        rollout=execution_target.ExecutionTarget.parse("jax:tpu"),
        inference=execution_target.ExecutionTarget.parse("jax:gpu"),
        placement=execution_target.Placement.DISAGGREGATED,
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    self.assertEqual(valid.weight_sync_mode, weight_sync.WeightSyncMode.RAIDEN)

    with self.assertRaisesRegex(
        ValueError,
        r"currently requires TPU accelerators on both trainer and rollout"
        r" \(GPU support for Raiden is not yet implemented\)",
    ):
      execution_target.ClusterTargets(
          trainer=execution_target.ExecutionTarget.parse("jax:gpu"),
          rollout=execution_target.ExecutionTarget.parse("jax:gpu:vllm"),
          inference=execution_target.ExecutionTarget.parse("jax:gpu"),
          placement=execution_target.Placement.DISAGGREGATED,
          weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
      )

  def test_weight_sync_mode_single_source_of_truth(self):
    self.assertIs(common_datatypes.WeightSyncMode, weight_sync.WeightSyncMode)
    self.assertIs(
        common_datatypes.WeightSyncMode, dist_datatypes.WeightSyncMode
    )
    self.assertEqual(
        common_datatypes.DEFAULT_WEIGHT_SYNC_MODE,
        weight_sync.DEFAULT_WEIGHT_SYNC_MODE,
    )

  def test_cluster_targets_validates_nccl_requires_gpu_endpoints(self):
    valid = execution_target.ClusterTargets(
        trainer=execution_target.ExecutionTarget.parse("jax:gpu"),
        rollout=execution_target.ExecutionTarget.parse("jax:gpu:vllm"),
        inference=execution_target.ExecutionTarget.parse("jax:tpu"),
        placement=execution_target.Placement.DISAGGREGATED,
        weight_sync_mode=weight_sync.WeightSyncMode.NCCL,
    )
    self.assertEqual(valid.weight_sync_mode, weight_sync.WeightSyncMode.NCCL)

    with self.assertRaisesRegex(ValueError, "nccl"):
      execution_target.ClusterTargets(
          trainer=execution_target.ExecutionTarget.parse("jax:tpu"),
          rollout=execution_target.ExecutionTarget.parse("jax:gpu:vllm"),
          inference=execution_target.ExecutionTarget.parse("jax:tpu"),
          placement=execution_target.Placement.DISAGGREGATED,
          weight_sync_mode=weight_sync.WeightSyncMode.NCCL,
      )

  def test_import_loads_no_ml_frameworks(self):
    forbidden = ("jax", "flax", "optax", "torch", "tpu_raiden")
    with mock.patch.dict(sys.modules):
      for name in list(sys.modules):
        if "tunix" in name.split("."):
          sys.modules.pop(name, None)
        elif any(pkg in name.split(".") for pkg in forbidden):
          sys.modules[name] = None
      for pkg in forbidden:
        sys.modules[pkg] = None
      mod = importlib.import_module("tunix.dist_rl.common.execution_target")
      self.assertTrue(hasattr(mod, "ExecutionTarget"))


if __name__ == "__main__":
  absltest.main()
