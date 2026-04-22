from __future__ import annotations

import random
from dataclasses import dataclass

from astrotsp.models.config import BenchmarkConfig


@dataclass(frozen=True)
class InstanceSpec:
    n_nodes: int
    instance_id: str
    body_ids: list[str]


def generate_instance_specs(config: BenchmarkConfig) -> list[InstanceSpec]:
    rng = random.Random(config.global_seed)
    specs: list[InstanceSpec] = []

    all_pool_ids: list[str] = []
    for pool in config.asteroid_pools:
        all_pool_ids.extend(str(x) for x in range(pool.start_id, pool.end_id + 1))

    if len(all_pool_ids) == 0:
        raise ValueError("No asteroid IDs available in configured pools.")

    anchor_count = len(config.anchor_body_ids)
    for n_nodes in config.n_values:
        if n_nodes <= anchor_count:
            raise ValueError("n_nodes must be greater than number of anchor bodies.")

        seen: set[tuple[str, ...]] = set()
        target_count = config.instances_per_n
        while len(seen) < target_count:
            draw_count = n_nodes - anchor_count
            sampled = rng.sample(all_pool_ids, k=draw_count)
            body_ids = [*config.anchor_body_ids, *sampled]
            key = tuple(body_ids)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                InstanceSpec(
                    n_nodes=n_nodes,
                    instance_id=f"n{n_nodes}_i{len(seen):02d}",
                    body_ids=body_ids,
                )
            )
    return specs
