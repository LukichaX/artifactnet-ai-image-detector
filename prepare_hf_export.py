from pathlib import Path
import torch
from safetensors.torch import save_file

ckpt_dir = Path("checkpoints")
checkpoint_path = ckpt_dir / "best_model_v5.pth"
if not checkpoint_path.exists():
    checkpoint_path = ckpt_dir / "best_model.pth"

export_dir = Path("hf_export")
export_dir.mkdir(exist_ok=True)

print(f"Loading checkpoint: {checkpoint_path} ...")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

clean_sd = {k: v.contiguous() for k, v in state_dict.items()}

safetensors_path = export_dir / "model.safetensors"
save_file(clean_sd, safetensors_path)

weights_path = export_dir / "artifactnet_v5_weights.pth"
torch.save(clean_sd, weights_path)

size_mb = safetensors_path.stat().st_size / (1024 ** 2)
print(f"Export complete! Size: {size_mb:.2f} MB -> saved to {export_dir}/")
