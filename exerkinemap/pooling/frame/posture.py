"""Compatibility re-exports for posture and activity adapters."""

from Exerkinetics.wearables.posture import (
    attach_deep_postures,
    attach_har_labels,
    attach_predicted_labels,
)

__all__ = ["attach_deep_postures", "attach_har_labels", "attach_predicted_labels"]