"""Thin Redis wrapper used as a parameter / coordination store.

Workers push gradient updates here between checkpoints; future Tenplex-style state
redistribution will use Redis as the rendezvous point during topology change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None


@dataclass
class RedisParamStore:
    url: str
    namespace: str = "tare"

    def __post_init__(self) -> None:
        if redis is None:
            raise RuntimeError("redis-py not installed; pip install redis")
        self.client = redis.Redis.from_url(self.url, decode_responses=True)

    def _key(self, *parts: str) -> str:
        return ":".join((self.namespace, *parts))

    def put_json(self, key: str, value) -> None:
        self.client.set(self._key(key), json.dumps(value))

    def get_json(self, key: str):
        raw = self.client.get(self._key(key))
        return json.loads(raw) if raw else None

    def put_blob(self, key: str, value: bytes) -> None:
        self.client.set(self._key(key), value)

    def get_blob(self, key: str) -> bytes | None:
        return self.client.get(self._key(key))
