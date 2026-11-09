from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "LukichaX/ArtifactNet-v5-ai-detector"

api = HfApi()
export_dir = Path("hf_export")

model_card = """---
language:
- en
license: mit
tags:
- image-classification
- ai-detection
- image-forensics
- synthetic-media
- pytorch
- from-scratch
pipeline_tag: image-classification
---

# ArtifactNet v5: From-Scratch Multi-Generator & Internet-Resilient AI vs. Real Detector

ArtifactNet v5 (~16.74M parameters) is a custom digital image forensics architecture trained **100% from scratch** (zero pretrained weights) across 65+ modern 2024–2026 synthetic generator architectures, real smartphone/DSLR camera corpora, and multi-hop social/messenger compression pipelines.

## Architectural Highlights
- **Stream 1 (Differential Micro-Residuals)**: NPR + Unbiased LPD + SRM + Zero-Sum L1-Bayar stem comparing Rich vs. Poor unclipped texture tiles within the same image.
- **Stream 2 (Wavelet & Spectral Phase Coherence)**: 3-level 2D Haar wavelet parent-child alignment + per-image normalized 2D-FFT amplitude & phase-gradient CNN.
- **Stream 3 (Meso-Structural & Self-Consistency ViT)**: 8 scale-free optical maps + 4 compression-survivable meso-gradient/shading maps + ConvNeXt stages + 3-layer 64-patch Pre-Norm Self-Consistency Transformer.
- **Stream 4 (Differential Gram Covariance & 28-Physics MLP)**: Centered cross-channel covariance differences + 28 `asinh`-stabilized dimensionless & meso-scale physical ratios.
- **12-Class Generator-Family Union Head + 16-Prototype Real-Camera Manifold**: Separates authentic camera modes on $\\mathbb{S}^{255}$ from 11 distinct synthetic generative physics families so unseen generators and re-compressed social downloads are caught reliably.

## Usage
```python
import torch
from safetensors.torch import load_file
from src.model import ArtifactNetV5
from src.forensic_council import ForensicCouncil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArtifactNetV5(dropout=0.0).to(device)
model.load_state_dict(load_file("model.safetensors"))
model.eval()

inspector = ForensicCouncil(model=model, device=device)
```
"""

if __name__ == "__main__":
    export_dir.mkdir(exist_ok=True)
    (export_dir / "README.md").write_text(model_card, encoding="utf-8")
    api.create_repo(repo_id=REPO_ID, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=REPO_ID)
    print(f"Uploaded {export_dir} to https://huggingface.co/{REPO_ID}")
