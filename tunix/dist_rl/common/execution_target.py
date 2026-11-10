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

"""Role-scoped execution target contracts for Framework x Accelerator x Engine.

Defines immutable, ML-framework-free configuration types describing which
framework (`jax` or `torch`), accelerator (`tpu` or `gpu`), and engine execute
each distributed RL role (`trainer`, `rollout`, `inference`), together with
fail-fast cross-role placement and weight-sync transport validation.
"""

from __future__ import annotations

import dataclasses
import enum

from tunix.common import datatypes as common_datatypes


class Framework(enum.StrEnum):
  """ML framework executing a role."""

  JAX = "jax"
  TORCH = "torch"


class Accelerator(enum.StrEnum):
  """Hardware accelerator executing a role."""

  TPU = "tpu"
  GPU = "gpu"


class Placement(enum.StrEnum):
  """Process/device placement mode between trainer and rollout roles."""

  COLOCATED = "colocated"
  DISAGGREGATED = "disaggregated"


_VALID_FRAMEWORKS = tuple(item.value for item in Framework)
_VALID_ACCELERATORS = tuple(item.value for item in Accelerator)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExecutionTarget:
  """Typed `(framework, accelerator, engine)` execution target for one role."""

  framework: Framework
  accelerator: Accelerator
  engine: str = ""

  def __post_init__(self) -> None:
    if ":" in self.engine or self.engine != self.engine.strip():
      raise ValueError(
          "ExecutionTarget.engine must not contain ':' or surrounding"
          f" whitespace, got {self.engine!r}."
      )

  @classmethod
  def parse(
      cls,
      spec: str,
      *,
      default_engine: str = "",
      allow_engine: bool = True,
  ) -> ExecutionTarget:
    """Parses `<framework>:<accelerator>[:<engine>]` into an `ExecutionTarget`.

    Args:
      spec: Target specification string, e.g. `"jax:tpu"`, `"jax:gpu"`, or
        `"torch:tpu:torchtitan"`.
      default_engine: Engine name used when `spec` omits `:<engine>`.
      allow_engine: Whether `:<engine>` is permitted in `spec` (set `False` for
        cluster-wide base shorthand `execution_targets.target`).

    Returns:
      Parsed `ExecutionTarget`.

    Raises:
      ValueError: If `spec` is malformed or names an unsupported framework or
        accelerator.
    """
    if not isinstance(spec, str) or not spec or spec != spec.strip():
      raise ValueError(
          "Execution target spec must be a non-empty string without"
          f" surrounding whitespace, got {spec!r}."
      )
    parts = spec.split(":")
    max_parts = 3 if allow_engine else 2
    if len(parts) < 2 or len(parts) > max_parts or any(not p for p in parts):
      expected = (
          "'<framework>:<accelerator>' or '<framework>:<accelerator>:<engine>'"
          if allow_engine
          else "'<framework>:<accelerator>'"
      )
      raise ValueError(
          f"Invalid execution target {spec!r}; expected {expected}."
      )
    if parts[0] not in _VALID_FRAMEWORKS:
      raise ValueError(
          f"Unsupported framework {parts[0]!r} in execution target {spec!r};"
          f" expected one of {_VALID_FRAMEWORKS}."
      )
    if parts[1] not in _VALID_ACCELERATORS:
      raise ValueError(
          f"Unsupported accelerator {parts[1]!r} in execution target {spec!r};"
          f" expected one of {_VALID_ACCELERATORS}."
      )
    engine = parts[2] if len(parts) == 3 else default_engine
    return cls(
        framework=Framework(parts[0]),
        accelerator=Accelerator(parts[1]),
        engine=engine,
    )

  def format(self) -> str:
    """Returns canonical `<framework>:<accelerator>[:<engine>]` string."""
    base = f"{self.framework.value}:{self.accelerator.value}"
    return f"{base}:{self.engine}" if self.engine else base


@dataclasses.dataclass(frozen=True, kw_only=True)
class ClusterTargets:
  """Resolved execution targets across trainer, rollout, and inference roles."""

  trainer: ExecutionTarget
  rollout: ExecutionTarget
  inference: ExecutionTarget
  placement: Placement = Placement.DISAGGREGATED
  weight_sync_mode: common_datatypes.WeightSyncMode = (
      common_datatypes.WeightSyncMode.NONE
  )

  def __post_init__(self) -> None:
    if self.placement == Placement.COLOCATED and (
        self.trainer.accelerator != self.rollout.accelerator
        or self.trainer.accelerator != self.inference.accelerator
    ):
      raise ValueError(
          "Colocated placement requires trainer, rollout, and inference to"
          f" share the same accelerator, got trainer={self.trainer.format()!r},"
          f" rollout={self.rollout.format()!r}, and"
          f" inference={self.inference.format()!r}."
      )
    if self.weight_sync_mode == common_datatypes.WeightSyncMode.RAIDEN and (
        self.trainer.accelerator != Accelerator.TPU
        or self.rollout.accelerator != Accelerator.TPU
    ):
      raise ValueError(
          "weight_sync_mode='raiden' currently requires TPU accelerators on"
          " both trainer and rollout (GPU support for Raiden is not yet"
          f" implemented), got trainer={self.trainer.format()!r} and"
          f" rollout={self.rollout.format()!r}."
      )
    if self.weight_sync_mode == common_datatypes.WeightSyncMode.NCCL and (
        self.trainer.accelerator != Accelerator.GPU
        or self.rollout.accelerator != Accelerator.GPU
    ):
      raise ValueError(
          "weight_sync_mode='nccl' requires GPU accelerators on both trainer"
          f" and rollout, got trainer={self.trainer.format()!r} and"
          f" rollout={self.rollout.format()!r}."
      )
