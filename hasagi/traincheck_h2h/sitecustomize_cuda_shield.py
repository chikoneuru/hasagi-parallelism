"""Machine-local shim for the TrainCheck venv: copy as ``sitecustomize.py``
into the venv's site-packages on hosts where any lazy CUDA query raises
(this torch build misreads the installed driver version, so even attribute
scans over ``torch.backends.cuda`` crash). TrainCheck's instrumentor
getattr-walks module attributes at setup time, before user code runs, so the
shield must be active at interpreter start. Converting the lazy plan-cache
lookup into AttributeError makes the scanner skip it; CPU-only runs never
need the cuFFT cache.
"""


def _shield() -> None:
    try:
        import torch.backends.cuda as bc
    except Exception:
        return

    def _no_lazy(self, name):
        raise AttributeError(name)

    try:
        type(bc.cufft_plan_cache).__getattr__ = _no_lazy
    except Exception:
        pass


_shield()


def _wrap_without_dump_dcp() -> None:
    """The collector's argument dumper cannot serialize the sharded-tensor
    objects that torch's distributed-checkpoint plumbing passes around (JSON
    dump fails, then the type() fallback trips ShardedTensor's torch-function
    guard). Keep those APIs wrapped but skip dumping their arguments."""
    try:
        import traincheck.config.config as tc_config
    except Exception:
        return
    for mod in ("torch.distributed.checkpoint", "torch.distributed._shard"):
        if mod not in tc_config.WRAP_WITHOUT_DUMP:
            tc_config.WRAP_WITHOUT_DUMP.append(mod)


_wrap_without_dump_dcp()


def _exception_safe_typename() -> None:
    """Legacy sharded-tensor objects intercept even type()/qualname probes via
    __torch_function__ and raise; any instrumented API that receives one then
    kills the collector inside typename(). Fall back to the plain class name
    instead of crashing; invariant semantics are unaffected (the object is
    recorded as an opaque value either way)."""
    try:
        import traincheck.instrumentor.dumper as td
        import traincheck.instrumentor.tracer as tt
        import traincheck.utils as tu
    except Exception:
        return
    orig = tu.typename

    def safe(o, *a, **k):
        try:
            return orig(o, *a, **k)
        except Exception:
            return f"unserializable.{o.__class__.__module__}.{o.__class__.__name__}"

    tu.typename = safe
    for mod in (td, tt):
        if getattr(mod, "typename", None) is orig:
            mod.typename = safe


_exception_safe_typename()


def _skip_dcp_internals() -> None:
    """Full instrumentation of torch's distributed-checkpoint internals makes
    the legacy sharded-tensor load path intractable (a 144-parameter load
    exceeded 40 minutes before timing out). Exclude those plumbing modules
    from instrumentation entirely; applied symmetrically to every arm and
    reference trace of the reshard scenario."""
    try:
        import traincheck.config.config as tc_config
    except Exception:
        return
    for mod in ("torch.distributed._shard", "torch.distributed.checkpoint",
                "torch.distributed._state_dict_utils", "torch.utils._pytree",
                "torch._ops", "torch._library", "torch.export"):
        if mod not in tc_config.INSTR_MODULES_TO_SKIP:
            tc_config.INSTR_MODULES_TO_SKIP.append(mod)


_skip_dcp_internals()
