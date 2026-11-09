import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import torch

from src.forensic_council import ForensicCouncil
from src.model import ArtifactNetV5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PROJECT_ROOT / "checkpoints"

_STATE = {
    "council": None,
    "ckpt_name": None,
    "ckpt_mtime": None,
    "epoch": None,
}


def _resolve_v5_ckpt() -> Path | None:
    v5_path = CKPT_DIR / "best_model_v5.pth"
    if v5_path.exists():
        return v5_path
    generic_path = CKPT_DIR / "best_model.pth"
    if generic_path.exists():
        try:
            meta = torch.load(generic_path, map_location="cpu", weights_only=False)
            if isinstance(meta, dict) and meta.get("architecture") == "ArtifactNetV5":
                return generic_path
        except Exception:
            pass
    last_v5 = CKPT_DIR / "last_model_v5.pth"
    if last_v5.exists():
        return last_v5
    return None


def get_council() -> tuple[ForensicCouncil | None, str]:
    ckpt_path = _resolve_v5_ckpt()
    if ckpt_path is None:
        return None, "untrained"

    mtime = ckpt_path.stat().st_mtime
    if _STATE["council"] is not None and _STATE["ckpt_name"] == ckpt_path.name and _STATE["ckpt_mtime"] == mtime:
        return _STATE["council"], f"{ckpt_path.name} (Epoch {_STATE['epoch']})"

    model = ArtifactNetV5(dropout=0.0).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("ema_state_dict", ckpt["model_state_dict"]) if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    council = ForensicCouncil(model=model, device=DEVICE)
    _STATE["council"] = council
    _STATE["ckpt_name"] = ckpt_path.name
    _STATE["ckpt_mtime"] = mtime
    _STATE["epoch"] = ckpt.get("epoch", "?")
    return council, f"{ckpt_path.name} (Epoch {_STATE['epoch']})"


get_council()


@torch.no_grad()
def analyze_image(
    img_rgb: np.ndarray, use_tta: bool = True
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, str]:
    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    if img_rgb is None:
        return (
            {"Real Photograph (Camera)": 0.0, "AI-Generated (Synthetic)": 0.0},
            empty,
            empty,
            empty,
            "Upload an image (fresh upload, messenger forward, social download, or screenshot) to run forensic inspection.",
        )

    council, status_str = get_council()
    if council is None:
        msg = (
            "### ⚠️ ArtifactNet v5 Checkpoint Not Trained Yet\n"
            "Run `python src/train.py` to train **ArtifactNet v5** from scratch. "
            "This dashboard hot-loads `checkpoints/best_model_v5.pth` automatically once Epoch 1 finishes."
        )
        return (
            {"Real Photograph (Camera)": 0.0, "AI-Generated (Synthetic)": 0.0},
            empty,
            empty,
            empty,
            msg,
        )

    res = council.analyze(img_rgb, use_tta=use_tta)
    ai_prob = res["ai_prob"]
    real_prob = res["real_prob"]

    conf_val = max(ai_prob, real_prob)
    if conf_val >= 0.88:
        tier = "High Confidence"
    elif conf_val >= 0.68:
        tier = "Moderate Confidence"
    else:
        tier = "Borderline / Mixed Signal"

    verdict_str = (
        f"🤖 **AI-Generated (Synthetic)** — `{ai_prob*100:.1f}% AI` vs `{real_prob*100:.1f}% Real` *({tier})*"
        if ai_prob >= 0.5
        else f"📷 **Real Photograph (Camera)** — `{real_prob*100:.1f}% Real` vs `{ai_prob*100:.1f}% AI` *({tier})*"
    )

    d_real = res["d_real"]
    tau = res["manifold_tau"]
    manifold_status = (
        f"Outside Camera Manifold (`d={d_real:.3f} > τ={tau:.3f}`)"
        if d_real > tau
        else f"Inside Camera Manifold (`d={d_real:.3f} <= τ={tau:.3f}`)"
    )

    w, h = res["orig_size"]
    color_mode = "Monochrome / Grayscale (Chroma Gate Active)" if res["is_monochrome"] else "Full RGB Spectrum"
    tta_str = (
        f"4-View Multi-Scale TTA (`{res['tta_agreement']*100:.0f}%` view agreement)"
        if use_tta
        else "Single Canonical View"
    )

    lines = [
        f"### 🎯 Verdict: {verdict_str}",
        f"- **Checkpoint:** `{status_str}` | **Inference Mode:** {tta_str}",
        f"- **Real-Camera Manifold (16 Prototypes):** {manifold_status}",
        f"- **Input Diagnostics:** `{w}×{h}px` | **Color Profile:** {color_mode} | **Stripped Border/Canvas:** `{res['canvas_ratio']*100:.1f}%`",
    ]

    if ai_prob >= 0.35 and res.get("top_families"):
        fam_str = " · ".join(f"**{fname}** (`{fprob*100:.0f}%`)" for fname, fprob in res["top_families"])
        lines.append(f"- **Closest Synthetic Architecture Signature:** {fam_str}")

    lines.extend(["", "### 🔬 Calibrated Multi-Stream & Physical Breakdown:"])
    for name, score, weight in res["experts"]:
        badge = "🔴 Synthetic Signal" if score >= 0.50 else "🟢 Camera Signal"
        lines.append(f"- **{name}:** `{score*100:.1f}% AI` ({badge}, weight `{weight:.2f}`)")

    return (
        {"AI-Generated (Synthetic)": ai_prob, "Real Photograph (Camera)": real_prob},
        res["scene_overlay"],
        res["micro_overlay"],
        res["xp_vis"],
        "\n".join(lines),
    )


with gr.Blocks(title="ArtifactNet v5 — Multi-Generator & Internet-Resilient Forensic Detector") as demo:
    gr.Markdown(
        "# 🔬 ArtifactNet v5: From-Scratch Multi-Generator & Internet-Resilient AI Image Detector\n"
        "100% custom architecture (`16.74M` params, **zero pretrained backbones**) trained from scratch across "
        "**65+ modern 2024–2026 generator families**, **real smartphone/DSLR corpora**, and **1-hop / 2-hop social & messenger compression chains**."
    )
    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(type="numpy", label="Upload Any Image (Fresh, Compressed, Downloaded, or Forwarded)")
            use_tta_chk = gr.Checkbox(
                value=True,
                label="Enable 4-View Multi-Scale + Flip Test-Time Augmentation (1024px Native + 640px Meso)",
            )
            btn = gr.Button("🔍 Run Forensic Inspection", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=2, label="Calibrated Forensic Verdict")
            out_md = gr.Markdown()

    with gr.Row():
        out_scene = gr.Image(label="Full-Scene Self-Consistency & Optical Anomaly Heatmap")
        out_micro = gr.Image(label="Rich-Tile 1:1 Mosaic + LPD/Bayar Micro-Residual Heatmap")
        out_poor = gr.Image(label="Smooth Unclipped Baseline Tiles (Sensor Floor Reference)")

    btn.click(
        fn=analyze_image,
        inputs=[in_img, use_tta_chk],
        outputs=[out_label, out_scene, out_micro, out_poor, out_md],
    )


if __name__ == "__main__":
    demo.launch()
