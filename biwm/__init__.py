"""BIWM 组件 registry (readme §11).

7 components:
1. canonical_belief.CanonicalBeliefWrapper      — strip harness decorations
2. blocked_action_log.BlockedActionLogWrapper   — expose blocked candidates+reasons
3. repair_unrolled.RepairUnrolledWrapper        — expand fail→repair→recover
4. verification_mask.VerificationMaskWrapper    — show verification_type/cost per step
5. shadow_execution.ShadowExecutionWrapper      — deterministic sandbox simulator
6. cross_harness_align.align_beliefs            — reduce N belief views → 1 aligned
7. cross_harness_align.self_consistency_score   — quantify disagreement as epistemic
"""
from .canonical_belief import CanonicalBeliefWrapper
from .blocked_action_log import BlockedActionLogWrapper
from .repair_unrolled import RepairUnrolledWrapper
from .verification_mask import VerificationMaskWrapper
from .shadow_execution import ShadowExecutionWrapper
from .cross_harness_align import align_beliefs, self_consistency_score


def biwm_full(inner):
    """BIWM-full = 1+2+3+4+5 stacked. inner = base harness (H0_raw etc.)."""
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
