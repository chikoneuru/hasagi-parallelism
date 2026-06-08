"""GPU Burst Pool Manager — abstracts worker lifecycle across backends."""
from tare.pool.worker_registry import WorkerInfo, WorkerRegistry

__all__ = ["WorkerInfo", "WorkerRegistry"]
