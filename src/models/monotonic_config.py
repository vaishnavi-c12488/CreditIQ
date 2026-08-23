"""
CreditIQ — Monotonic Constraint Configuration

Only features with a clear and defensible risk direction are constrained.
All other features remain unconstrained.
"""

MONOTONIC_CONSTRAINTS = {
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
}