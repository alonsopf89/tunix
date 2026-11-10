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

"""Runtime bootstrap plugin for `torch:tpu` worker processes (`torchtpu`)."""

from __future__ import annotations

import importlib
import os

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.distributed.runtime.bootstraps import bootstrap_api


@bootstrap_api.BOOTSTRAP_REGISTRY.register("torch:tpu")
class TorchTpuBootstrap(bootstrap_api.RuntimeBootstrap):
  """Initializes PyTorch PJRT TPU runtime for `torchtpu`."""

  framework = execution_target.Framework.TORCH
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
    del context, weight_sync_mode
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if not initialize_runtime:
      return
    importlib.import_module("torch")
    raw_tpu_module = (
        os.environ["TORCH_TPU_MODULE"].strip()
        if "TORCH_TPU_MODULE" in os.environ
        else ""
    )
    importlib.import_module(raw_tpu_module or "torchtpu")
