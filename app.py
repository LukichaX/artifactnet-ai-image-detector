import sys
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import get_transforms
from model import ArtifactNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
IMG_SIZE = checkpoint.get("config", {}).get("img_size", 256)

MODEL = ArtifactNet(num_classes=1, dropout_rate=0.0).to(DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()

_, VAL_TRANSFORM = get_transforms(img_size=IMG_SIZE)


def generate_gradcam_and_predict(image_rgb: np.ndarray):
    """
    ითვლის მოდელის პროგნოზს (Real vs AI), აგენერირებს Grad-CAM სითბურ რუკას
    და აჩვენებს High-Pass სიხშირულ ფილტრს.
    """
    if image_rgb is None:
        return {}, None, None

    orig_h, orig_w = image_rgb.shape[:2]
    tensor_img = VAL_TRANSFORM(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

    # ვიჭერთ ბოლო კონვოლუციური შრის (stage4) აქტივაციებს და გრადიენტებს
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    handle_f = MODEL.stage4.register_forward_hook(forward_hook)
    handle_b = MODEL.stage4.register_full_backward_hook(backward_hook)

    try:
        MODEL.zero_grad()
        logits = MODEL(tensor_img)
        prob_ai = torch.sigmoid(logits).item()
        prob_real = 1.0 - prob_ai

        # თუ AI-ის ალბათობა მეტია, ვხატავთ რამ გამოიწვია AI-ის სიგნალი; თუ Real — რამ გამოიწვია Real
        score = logits if prob_ai >= 0.5 else -logits
        score.backward()

        acts = activations[0].detach().cpu().numpy()[0]   # (512, H, W)
        grads = gradients[0].detach().cpu().numpy()[0]    # (512, H, W)

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_rgb, 0.55, heatmap, 0.45, 0)

        # ასევე გამოვჭრათ High-Pass სიხშირული ანაბეჭდი ვიზუალიზაციისთვის
        with torch.no_grad():
            hp_feat = MODEL.freq_extractor(tensor_img).abs().mean(dim=1).squeeze().cpu().numpy()
            if hp_feat.max() > 0:
                hp_feat = hp_feat / hp_feat.max()
            hp_vis = cv2.resize(hp_feat, (orig_w, orig_h))
            hp_vis = cv2.applyColorMap(np.uint8(255 * hp_vis), cv2.COLORMAP_VIRIDIS)
            hp_vis = cv2.cvtColor(hp_vis, cv2.COLOR_BGR2RGB)

    finally:
        handle_f.remove()
        handle_b.remove()

    confidences = {
        "🤖 AI-Generated (ხელოვნური)": float(prob_ai),
        "📷 Real Photo (რეალური)": float(prob_real),
    }

    return confidences, overlay, hp_vis


with gr.Blocks(title="ArtifactNet — AI vs Real Image Detector") as demo:
    gr.Markdown(
        """
        # 🛡️ ArtifactNet: AI-Generated vs Real Image Detector
        **ნულიდან შექმნილი ღრმა ნეირონული ქსელი (18.9M პარამეტრი | Validation F1: 98.94% | ROC-AUC: 99.92%)**
        ჩააგდე ნებისმიერი ფოტო ქვემოთ — მოდელი დაითვლის ალბათობას და **Grad-CAM სითბური რუკით** გაჩვენებს, სურათის რომელ უბნებს დაეყრდნო გადაწყვეტილების მიღებისას.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="ატვირთე სურათი შესამოწმებლად")
            analyze_btn = gr.Button("🔍 სურათის ანალიზი", variant="primary")
            label_out = gr.Label(num_top_classes=2, label="მოდელის ვერდიქტი")

        with gr.Column(scale=1):
            gradcam_out = gr.Image(label="🔥 Grad-CAM სითბური რუკა (წითელი = მთავარი ფოკუსის ზონა)")
            freq_out = gr.Image(label="⚡ High-Pass სიხშირული არტეფაქტების რუკა")

    analyze_btn.click(
        fn=generate_gradcam_and_predict,
        inputs=input_img,
        outputs=[label_out, gradcam_out, freq_out],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)