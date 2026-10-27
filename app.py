import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import torch

from src.dataset import smash_and_reconstruct_rich_poor
from src.model import ArtifactNetV2

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_V2 = PROJECT_ROOT / "checkpoints" / "best_model_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ArtifactNetV2(dropout=0.0).to(DEVICE)
if CKPT_V2.exists():
    ckpt = torch.load(CKPT_V2, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
model.eval()


@torch.no_grad()
def analyze_image(img_rgb: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
    if img_rgb is None:
        return {"Real Photograph": 0.0, "AI-Generated": 0.0}, np.zeros((256, 256, 3), dtype=np.uint8), np.zeros((256, 256, 3), dtype=np.uint8), "ატვირთე სურათი."

    # 2-ჯერადი სემპლინგი 1:1 პიქსელებზე (TTA) მაღალი სიზუსტისთვის
    probs_list, dist_list, rad_list = [], [], []
    xr_vis, xp_vis = smash_and_reconstruct_rich_poor(img_rgb, out_size=256, patch_size=32, is_train=False)

    for flip_code in [None, 1]:
        work_img = img_rgb if flip_code is None else cv2.flip(img_rgb, flip_code)
        xr_np, xp_np = smash_and_reconstruct_rich_poor(work_img, out_size=256, patch_size=32, is_train=False)
        xr = torch.from_numpy(xr_np.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0
        xp = torch.from_numpy(xp_np.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit, dist_sq, radius_sq = model(xr, xp, return_aux=True)
            prob = torch.sigmoid(logit).item()

        probs_list.append(prob)
        dist_list.append(float(dist_sq.item()))
        rad_list.append(float(radius_sq.item()))

    ai_prob = float(np.mean(probs_list))
    real_prob = 1.0 - ai_prob
    mean_dist = float(np.mean(dist_list))
    mean_rad = float(np.mean(rad_list))

    # ვქმნით LPD + NPR ფორენზიკული რენტგენის ვიზუალიზაციას Rich ტექსტურაზე
    xr_tensor = torch.from_numpy(xr_vis.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0
    lpd_map = model.frontend.lpd(xr_tensor)[0, 3:6].mean(dim=0).cpu().numpy()
    lpd_norm = np.clip(lpd_map / (np.percentile(lpd_map, 98) + 1e-6), 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap((lpd_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rich = cv2.addWeighted(xr_vis, 0.45, heatmap_rgb, 0.55, 0)

    sphere_status = "⚠️ სფეროს გარეთ (AI / სინთეტიკური ანომალია)" if mean_dist > mean_rad else "✅ რეალური კამერის სფეროს შიგნით"
    report = (
        f"**ფორენზიკული დიაგნოზი (1:1 Native Pixel Analysis):**\n"
        f"- **სურათის ორიგინალი რეზოლუცია:** `{img_rgb.shape[1]}x{img_rgb.shape[0]}` (შემოწმდა Resize-ით დაჭყლეტის გარეშე!)\n"
        f"- **მანძილი რეალური კამერის ცენტრიდან (`||z - c_real||²`):** `{mean_dist:.4f}` (საზღვრის რადიუსი `R² = {mean_rad:.4f}`)\n"
        f"- **ჰიპერსფეროს სტატუსი:** {sphere_status}"
    )

    return (
        {"AI-Generated (Synthetic)": ai_prob, "Real Photograph (Camera)": real_prob},
        overlay_rich,
        xp_vis,
        report,
    )


with gr.Blocks(title="ArtifactNet v2.5 Ultra — Forensic AI Detector") as demo:
    gr.Markdown(
        "# 🔬 ArtifactNet v2.5 Ultra: 6-Pillar Forensic & Hypersphere AI Image Detector (100% From Scratch)\n"
        "ამოწმებს სურათს **1:1 ორიგინალ პიქსელებში** (PatchCraft Rich/Poor ტექსტურული კონტრასტით, "
        "FerretNet LPD-ით, CVPR 2024 NPR-ით, AAAI 2024 FreqNet-ითა და SimLBR ჰიპერსფერული საზღვრით)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(type="numpy", label="ატვირთე სურათი (ნებისმიერი რეზოლუციის)")
            btn = gr.Button("🔍 ფორენზიკული სკანირება", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=2, label="კლასიფიკაციის შედეგი")
            out_md = gr.Markdown()
    with gr.Row():
        out_rich = gr.Image(label="Rich Texture 1:1 Tiles + FerretNet LPD Heatmap")
        out_poor = gr.Image(label="Poor Texture 1:1 Tiles (Smooth Regions Baseline)")

    btn.click(fn=analyze_image, inputs=in_img, outputs=[out_label, out_rich, out_poor, out_md])

if __name__ == "__main__":
    demo.launch()