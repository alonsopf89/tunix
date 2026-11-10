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

"""Runtime bootstrap plugin for `jax:tpu` worker processes."""

from __future__ import annotations

import os

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.distributed.runtime.bootstraps import bootstrap_api


@bootstrap_api.BOOTSTRAP_REGISTRY.register("jax:tpu")
class JaxTpuBootstrap(bootstrap_api.RuntimeBootstrap):
  """Initializes TPU Raiden preload (when enabled) and JAX TPU runtime."""

  framework = execution_target.Framework.JAX
  accelerator = execution_target.Accelerator.TPU

  def initialize(
      self,
      context: context_lib.ProcessContext | None = None,
      *,
      weight_sync_mode: datatypes.WeightSyncMode = (
          datatypes.WeightSyncMode.NONE
      ),
      initialize_runtime: bool = True,
  ) -> None:
    use_raiden_env = "USE_RAIDEN" in os.environ and os.environ[
        "USE_RAIDEN"
    ].strip().lower() in ("1", "true", "yes")
    if use_raiden_env or weight_sync_mode == datatypes.WeightSyncMode.RAIDEN:
      from tunix.dist_rl.weight_sync import raiden_preload  # pylint: disable=g-import-not-at-top

      raiden_preload.import_raiden()
    if initialize_runtime and context is not None:
      context.jax.initialize()
