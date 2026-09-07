"""Current release claims, separate from historical benchmark artifacts."""

from .version import __version__


def release_claims() -> dict:
    return {
        "schema_version": "rival.release-claims.v1",
        "release": __version__,
        "mode": "research",
        "offering": "supervised concept and message scenario comparisons",
        "estimand": "aggregate choice shares for a declared audience and scenario",
        "decision_support_qualified": False,
        "confidence_qualified": False,
        "individual_fidelity_qualified": False,
        "customer_launch_ready": False,
        "required_evidence": "untouched relevant studies, classical baselines, coverage and operating measurements",
        "limitations": [
            "Simulation draws are not independent human participants.",
            "The bundled demo uses generated people, outcomes and anchors.",
            "Historical benchmark results do not qualify the current release or a new audience.",
        ],
    }
