import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import gradio as gr
import numpy as np
import torch

from src.forensic_council import ForensicCouncil
from src.model import ArtifactNetV2

CKPT_V2 = PROJECT_ROOT / "checkpoints" / "best_model_v2.pth"
CKPT_FALLBACK = PROJECT_ROOT / "checkpoints" / "best_model.pth"
CKPT_PATH = CKPT_V2 if CKPT_V2.exists() else CKPT_FALLBACK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ArtifactNetV2(dropout=0.0).to(DEVICE)
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

council = ForensicCouncil(model=model, device=DEVICE)


@torch.no_grad()
def analyze_image(img_rgb: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
    if img_rgb is None:
        empty = np.zeros((256, 256, 3), dtype=np.uint8)
        return {"Real Photograph (Camera)": 0.0, "AI-Generated (Synthetic)": 0.0}, empty, empty, "ატვირთე სურათი."

    res = council.analyze(img_rgb)
    ai_prob = res["ai_prob"]
    real_prob = res["real_prob"]
    xr_vis = res["xr_vis"]
    xp_vis = res["xp_vis"]

    # LPD ფორენზიკული სითბური რუკა Rich ტექსტურაზე
    xr_tensor = torch.from_numpy(np.ascontiguousarray(xr_vis.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0
    lpd_map = model.frontend.lpd(xr_tensor)[0, 3:6].mean(dim=0).cpu().numpy()
    lpd_norm = np.clip(lpd_map / (np.percentile(lpd_map, 98) + 1e-6), 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap((lpd_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rich = cv2.addWeighted(xr_vis, 0.45, heatmap_rgb, 0.55, 0)

    verdict_str = (
        f"🤖 **AI-Generated (ხელოვნური)** — `{ai_prob*100:.1f}% AI` vs `{real_prob*100:.1f}% Real`"
        if ai_prob >= 0.5
        else f"📷 **Real Photograph (რეალური ფოტო)** — `{real_prob*100:.1f}% Real` vs `{ai_prob*100:.1f}% AI`"
    )

    bw_str = "კი (მონოქრომული/ვინტაჟური ფილტრი გააქტიურდა)" if res["is_monochrome"] else "არა (სრული ფერადი სპექტრი)"
    w, h = res["orig_size"]

    lines = [
        f"### 🎯 ვერდიქტი: {verdict_str}",
        f"- **კონსილიუმის კვორუმი:** `{res['flagged_count']} / {res['total_experts']}` ექსპერტი აფიქსირებს AI ანომალიას *(AI ვერდიქტისთვის საჭიროა >= 6 ფლაგი)*",
        f"- **ორიგინალი რეზოლუცია:** `{w}x{h}` | **შავ-თეთრი რეჟიმი:** `{bw_str}` | **მოჭრილი ჩარჩო/კანვასი:** `{res['canvas_ratio']*100:.1f}%`",
        "",
        "### 🏛️ 15 სპეციალიზებული ფორენზიკული ექსპერტის დასკვნა:",
    ]

    for name, score, _ in res["experts"]:
        is_ai_flag = score >= 0.55
        badge = "🔴 AI" if is_ai_flag else "🟢 Real"
        lines.append(f"- **{name}:** `{score*100:.1f}% AI` ({badge})")

    report = "\n".join(lines)

    return (
        {"AI-Generated (Synthetic)": ai_prob, "Real Photograph (Camera)": real_prob},
        overlay_rich,
        xp_vis,
        report,
    )


with gr.Blocks(title="ArtifactNet v3.5 — 15-Expert Forensic Council") as demo:
    gr.Markdown(
        "# 🏛️ ArtifactNet v3.5: 15-Expert Forensic Council (AI vs. Real Detector)\n"
        "სურათს აანალიზებს **15 დამოუკიდებელი სპეციალისტი** (3 ღრმა ნეირონული შტო + 12 ოპტიკური, სენსორული, "
        "ფოტონური ხმაურის, ქრომატული აბერაციისა და სიხშირული ფიზიკის ექსპერტი) კვორუმის პრინციპით."
    )
    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(type="numpy", label="ატვირთე სურათი (ნებისმიერი რეზოლუციის)")
            btn = gr.Button("🔍 კონსილიუმის სკანირება (15 Experts)", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=2, label="კონსილიუმის საბოლოო ვერდიქტი")
            out_md = gr.Markdown()
    with gr.Row():
        out_rich = gr.Image(label="Rich Texture 1:1 Tiles + FerretNet LPD Heatmap (ჩარჩო-გაფილტრული)")
        out_poor = gr.Image(label="Poor Texture 1:1 Tiles (Smooth Regions Baseline)")

    btn.click(fn=analyze_image, inputs=in_img, outputs=[out_label, out_rich, out_poor, out_md])

if __name__ == "__main__":
    demo.launch()