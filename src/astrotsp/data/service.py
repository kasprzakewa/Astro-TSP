from __future__ import annotations

from datetime import datetime

from astrotsp.data.horizons import HorizonsClient
from astrotsp.models.costs import build_instance
from astrotsp.models.problem import TSPInstance


class InstanceBuilder:
    def __init__(self, horizons_client: HorizonsClient | None = None) -> None:
        self.horizons_client = horizons_client or HorizonsClient()

    def from_horizons(self, body_ids: list[str], epoch: datetime) -> TSPInstance:
        if len(set(body_ids)) != len(body_ids):
            raise ValueError("Duplicate body IDs are not allowed.")
        nodes = self.horizons_client.fetch_bodies(body_ids=body_ids, epoch=epoch)
        return build_instance(nodes=nodes, epoch=epoch)
