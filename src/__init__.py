"""
ArtifactNet v5: From-Scratch Multi-Generator & Internet-Resilient Forensic Detector.
"""
from src.model import ArtifactNet, ArtifactNetV2, ArtifactNetV4, ArtifactNetV5, FAMILY_NAMES
from src.forensic_council import ForensicCouncil, ForensicInspectorV4, ForensicInspectorV5

__all__ = [
    "ArtifactNet",
    "ArtifactNetV2",
    "ArtifactNetV4",
    "ArtifactNetV5",
    "FAMILY_NAMES",
    "ForensicCouncil",
    "ForensicInspectorV4",
    "ForensicInspectorV5",
]
