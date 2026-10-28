# Copyright 2026 The Tunix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the engine with its KV caches split into data parallel ranks."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

from tunix.experimental.generate import testing_utils
from tunix.experimental.generate import utils

# The DP ranks need 4 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

_NUM_RANKS = 4

# (request id, prompt, route key) of each request.
_REQUESTS = (
    ("a", [3, 5, 7], 0),
    ("b", [2, 4], 1),
    ("c", [9, 1, 1, 2, 3, 4, 5], 1),
    ("d", [6, 6, 6, 6, 6, 6, 6, 6, 6], 3),
    ("e", [8, 2, 2], None),
    ("f", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 5),
)


def _dp_mesh() -> jax.sharding.Mesh:
  return jax.sharding.Mesh(
      np.array(jax.devices("cpu")[:_NUM_RANKS]), ("dp",)
  )


def _add_requests(engine, requests=_REQUESTS):
  for request_id, prompt, route_key in requests:
    request = testing_utils.make_request(
        request_id, prompt, route_key=route_key
    )
    engine.add_request(request)


def _run(engine) -> dict[str, list[int]]:
  outputs = {}
  for _ in range(200):
    if not engine.has_unfinished_requests():
      break
    for output in engine.step():
      outputs[output.request_id] = output.token_ids.tolist()
  return outputs


class EngineDPTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < _NUM_RANKS:
      self.skipTest("needs 4 CPU devices")

  @parameterized.parameters(1, 2)
  def test_matches_a_single_rank(self, num_scheduler_steps):
    single = testing_utils.make_engine(num_scheduler_steps=num_scheduler_steps)
    _add_requests(single)
    expected = _run(single)

    engine = testing_utils.make_engine(
        num_scheduler_steps=num_scheduler_steps, dp_mesh=_dp_mesh()
    )
    _add_requests(engine)
    outputs = _run(engine)

    self.assertLen(expected, len(_REQUESTS))
    self.assertEqual(outputs, expected)

  def test_requests_with_the_same_route_key_share_a_rank(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())
    _add_requests(engine)

    engine.step()

    ranks = {
        request_id: engine._request_to_rank[request_id]
        for request_id, _, _ in _REQUESTS
    }
    self.assertEqual(ranks["a"], 0)
    self.assertEqual(ranks["b"], 1)
    self.assertEqual(ranks["c"], 1)
    self.assertEqual(ranks["d"], 3)
    # Route key 5 lands in rank 5 % 4.
    self.assertEqual(ranks["f"], 1)

  def test_each_rank_holds_its_own_pages(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())
    _add_requests(engine)

    engine.step()

    pages = utils.stitch_physical_pages(
        engine.mesh,
        [c.get_physical_pages() for c in engine._dp_coordinators],
    )["layer_0"]
    # One rank's pages on each device.
    self.assertEqual(
        pages.sharding.shard_shape(pages.shape)[0],
        pages.shape[0] // _NUM_RANKS,
    )
    for coordinator in engine._dp_coordinators:
      self.assertFalse(
          coordinator.get_physical_pages()["layer_0"].is_deleted()
      )

  def test_each_rank_has_its_own_dp_coordinator(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())

    self.assertLen(engine._dp_coordinators, _NUM_RANKS)

  def test_rank_pages_are_a_block_of_the_pages_on_its_device(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())
    pages = utils.stitch_physical_pages(
        engine.mesh,
        [c.get_physical_pages() for c in engine._dp_coordinators],
    )["layer_0"]
    num_rank_pages = pages.shape[0] // _NUM_RANKS

    for r, coordinator in enumerate(engine._dp_coordinators):
      (rank_shard,) = coordinator.get_physical_pages()[
          "layer_0"
      ].addressable_shards
      (shard,) = [
          shard
          for shard in pages.addressable_shards
          if shard.device == rank_shard.device
      ]
      self.assertEqual(
          shard.index[0],
          slice(r * num_rank_pages, (r + 1) * num_rank_pages),
      )
      # The rank's host transfers address the memory it allocated.
      self.assertEqual(
          shard.data.unsafe_buffer_pointer(),
          rank_shard.data.unsafe_buffer_pointer(),
      )

  def test_routes_by_route_key(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())

    for i in range(6):
      engine.add_request(testing_utils.make_request(f"r{i}", [1], route_key=i))

    self.assertEqual(
        [engine._request_to_rank[f"r{i}"] for i in range(6)],
        [0, 1, 2, 3, 0, 1],
    )

  def test_requests_without_a_route_key_hash_their_id(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())

    engine.add_request(testing_utils.make_request("some_request", [1]))

    self.assertEqual(
        engine._request_to_rank["some_request"],
        utils.stable_route_hash("some_request") % _NUM_RANKS,
    )

  def test_negative_int_route_key_raises_and_queues_nothing(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())

    with self.assertRaises(ValueError):
      engine.add_request(testing_utils.make_request("r", [1], route_key=-1))

    self.assertEmpty(engine._request_to_rank)
    self.assertFalse(engine.has_unfinished_requests())

  def test_batch_holds_each_ranks_rows_back_to_back(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())
    prompt = [1, 2, 3, 4]
    _add_requests(
        engine, [("g2", prompt, 2), ("g0", prompt, 0), ("g2b", prompt, 6)]
    )

    batch = engine._schedule_step()

    self.assertEqual(
        [r.request_id for r in batch.requests], ["g0", "g2", "g2b"]
    )
    self.assertEqual(
        batch.distributions, ((0, 0, 1), (0, 0, 0), (0, 0, 2), (0, 0, 0))
    )
    self.assertEqual(engine._max_num_rows, _NUM_RANKS * 4)

  def test_each_rank_has_the_full_budgets(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh(), max_num_seqs=2)
    # Three requests for rank 0 and one for rank 1.
    _add_requests(
        engine,
        [(f"g0_{i}", [1, 2, 3, 4], 0) for i in range(3)]
        + [("g1", [1, 2, 3, 4], 1)],
    )

    batch = engine._schedule_step()

    # Rank 0 runs its first two requests, and its third waits without holding
    # up rank 1.
    self.assertEqual(
        [r.request_id for r in batch.requests], ["g0_0", "g0_1", "g1"]
    )

  def test_finished_requests_complete_independently_per_rank(self):
    engine = testing_utils.make_engine(dp_mesh=_dp_mesh())
    engine.add_request(
        testing_utils.make_request("r0", [1], route_key=0, max_tokens=1)
    )
    engine.add_request(testing_utils.make_request("r1", [1], route_key=1))

    outputs = engine.step()

    self.assertEqual([output.request_id for output in outputs], ["r0"])
    self.assertEqual(
        [c.has_unfinished_requests() for c in engine._dp_coordinators],
        [False, True, False, False],
    )


if __name__ == "__main__":
  absltest.main()
