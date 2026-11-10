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

"""Runtime bootstrap plugin for `jax:gpu` worker processes."""

from __future__ import annotations

import os

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.distributed.runtime.bootstraps import bootstrap_api


@bootstrap_api.BOOTSTRAP_REGISTRY.register("jax:gpu")
class JaxGpuBootstrap(bootstrap_api.RuntimeBootstrap):
  """Initializes CUDA/XLA environment defaults and JAX GPU distributed state."""

  framework = execution_target.Framework.JAX
  accelerator = execution_target.Accelerator.GPU

  def initialize(
      self,
      context: context_lib.ProcessContext | None = None,
      *,
      weight_sync_mode: datatypes.WeightSyncMode = (
          datatypes.WeightSyncMode.NONE
      ),
      initialize_runtime: bool = True,
  ) -> None:
    del weight_sync_mode
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    if initialize_runtime and context is not None:
      context.jax.initialize()
