from .canonical_belief import CanonicalBeliefWrapper
from .blocked_action_log import BlockedActionLogWrapper
from .repair_unrolled import RepairUnrolledWrapper
from .verification_mask import VerificationMaskWrapper
from .shadow_execution import ShadowExecutionWrapper
from .cross_harness_align import align_beliefs, self_consistency_score


def biwm_full(inner):
    h = inner
    h = CanonicalBeliefWrapper(h)
    h = BlockedActionLogWrapper(h)
    h = RepairUnrolledWrapper(h)
    h = VerificationMaskWrapper(h)
    h = ShadowExecutionWrapper(h)
    h.harness_id = f"BIWMfull_{inner.harness_id}"
    return h


BIWM_WRAPPERS = {
    "canonical": CanonicalBeliefWrapper,
    "blocked_action_log": BlockedActionLogWrapper,
    "repair_unrolled": RepairUnrolledWrapper,
    "verification_mask": VerificationMaskWrapper,
    "shadow_execution": ShadowExecutionWrapper,
}

__all__ = [
    "CanonicalBeliefWrapper",
    "BlockedActionLogWrapper",
    "RepairUnrolledWrapper",
    "VerificationMaskWrapper",
    "ShadowExecutionWrapper",
    "align_beliefs",
    "self_consistency_score",
    "biwm_full",
    "BIWM_WRAPPERS",
]
