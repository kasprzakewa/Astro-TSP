from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from astropy.time import Time
from astroquery.jplhorizons import Horizons

from astrotsp.models.problem import CelestialBody


@dataclass(frozen=True)
class BodyRecord:
    name: str
    horizons_id: str
    x: float
    y: float
    z: float
    epoch: str


class HorizonsClient:
    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bodies(self, body_ids: list[str], epoch: datetime) -> list[CelestialBody]:
        if not body_ids:
            raise ValueError("body_ids must not be empty.")

        epoch_iso = epoch.strftime("%Y-%m-%d %H:%M:%S")
        records: list[BodyRecord] = []
        for body_id in body_ids:
            records.append(self._fetch_or_load(body_id=body_id, epoch_iso=epoch_iso))

        return [
            CelestialBody(
                name=r.name,
                horizons_id=r.horizons_id,
                x=r.x,
                y=r.y,
                z=r.z,
            )
            for r in records
        ]

    def _fetch_or_load(self, body_id: str, epoch_iso: str) -> BodyRecord:
        cache_file = self.cache_dir / f"{body_id}_{epoch_iso.replace(' ', 'T')}.json"
        if cache_file.exists():
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            return BodyRecord(**loaded)

        epoch_jd = Time(epoch_iso, format="iso", scale="tdb").jd
        obj = Horizons(id=body_id, location="@sun", epochs=epoch_jd)
        vectors = obj.vectors()
        if len(vectors) == 0:
            raise ValueError(f"No vectors returned for body id '{body_id}'.")

        row = vectors[0]
        record = BodyRecord(
            name=str(row["targetname"]),
            horizons_id=body_id,
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            epoch=epoch_iso,
        )
        cache_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        return record
