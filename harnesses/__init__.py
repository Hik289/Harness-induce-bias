"""Harness registry. H0-H5 全部 Day-2 真实实现; H6 BIWM 留 Day 5."""
from .h0_raw import H0RawHarness
from .h1_structured import H1StructuredHarness
from .h2_risk_gated import H2RiskGatedHarness
from .h3_repair_heavy import H3RepairHeavyHarness
from .h4_verification_selective import H4VerificationSelectiveHarness
from .h5_cost_aware import H5CostAwareHarness


HARNESS_REGISTRY = {
    "H0_raw": H0RawHarness,
    "H1_structured": H1StructuredHarness,
    "H2_risk_gated": H2RiskGatedHarness,
    "H3_repair_heavy": H3RepairHeavyHarness,
    "H4_verification_selective": H4VerificationSelectiveHarness,
    "H5_cost_aware": H5CostAwareHarness,
}


__all__ = [
    "H0RawHarness",
    "H1StructuredHarness",
    "H2RiskGatedHarness",
    "H3RepairHeavyHarness",
    "H4VerificationSelectiveHarness",
    "H5CostAwareHarness",
    "HARNESS_REGISTRY",
]
