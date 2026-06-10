"""attest: an equivalence certificate for elastic-reconfiguration transitions.

When a distributed training job moves between two layouts (a world-size change,
DDP <-> FSDP rewrap, or an optimizer-state reshard), the state transport is
usually validated only post-hoc, by watching the loss curve. This package
checks the transition itself: it snapshots the logical training state on both
sides of the reshard, certifies their equivalence against explicit invariants,
and gates the commit so a violated transition rolls back to the last verified
state instead of training on silently corrupted weights.

The reshard mechanism (torch.distributed.checkpoint or any other) is treated
as a black box under test; the certificate is computed from the state itself.
"""

from .certificate import TransitionCertificate, Violation
from .gate import CommitDecision, certify_transition
from .snapshot import StateSnapshot, snapshot_from_state_dicts

__all__ = [
    "CommitDecision",
    "StateSnapshot",
    "TransitionCertificate",
    "Violation",
    "certify_transition",
    "snapshot_from_state_dicts",
]
