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

"""Framework x Accelerator runtime bootstrap contract and lazy plugin registry."""

from __future__ import annotations

import abc
from typing import ClassVar

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.common import registry
from tunix.dist_rl.distributed.runtime import context as context_lib

WeightSyncMode = datatypes.WeightSyncMode

BOOTSTRAP_REGISTRY = registry.Registry("RuntimeBootstrapRegistry")

_BUILTIN_BOOTSTRAP_MODULES: dict[str, str] = {
    "jax:tpu": "tunix.dist_rl.distributed.runtime.bootstraps.jax_tpu",
    "jax:gpu": "tunix.dist_rl.distributed.runtime.bootstraps.jax_gpu",
    "torch:tpu": "tunix.dist_rl.distributed.runtime.bootstraps.torch_tpu",
}


def bootstrap_key(
    framework: execution_target.Framework,
    accelerator: execution_target.Accelerator,
) -> str:
  """Returns the canonical `<framework>:<accelerator>` registry key."""
  return f"{framework.value}:{accelerator.value}"


class RuntimeBootstrap(abc.ABC):
  """Abstract lifecycle bootstrap for one `(Framework, Accelerator)` cell."""

  framework: ClassVar[execution_target.Framework]
  accelerator: ClassVar[execution_target.Accelerator]

  @abc.abstractmethod
  def initialize(
      self,
      context: context_lib.ProcessContext | None = None,
      *,
      weight_sync_mode: datatypes.WeightSyncMode = (
          datatypes.WeightSyncMode.NONE
      ),
      initialize_runtime: bool = True,
  ) -> None:
    """Initializes process-level framework and accelerator runtime state.

    Args:
      context: Optional distributed `ProcessContext` providing cluster
        rendezvous and discovery hooks.
      weight_sync_mode: Configured cluster weight synchronization transport.
      initialize_runtime: When `False`, prepares environment/preload invariants
        while deferring distributed runtime initialization to a downstream
        engine (e.g., external `vLLM` worker processes).
    """


def get_runtime_bootstrap(
    target: execution_target.ExecutionTarget,
) -> RuntimeBootstrap:
  """Resolves and instantiates the `RuntimeBootstrap` for `target`."""
  key = bootstrap_key(target.framework, target.accelerator)
  if not BOOTSTRAP_REGISTRY.contains(key) and key in _BUILTIN_BOOTSTRAP_MODULES:
    __import__(_BUILTIN_BOOTSTRAP_MODULES[key], fromlist=["*"])
  bootstrap_cls = BOOTSTRAP_REGISTRY.get(key)
  if not issubclass(bootstrap_cls, RuntimeBootstrap):
    raise TypeError(
        f"Registered bootstrap {bootstrap_cls!r} for {key!r} must subclass"
        " RuntimeBootstrap."
    )
  return bootstrap_cls()
