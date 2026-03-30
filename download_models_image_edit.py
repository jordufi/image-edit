import os
import shutil
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

os.environ["HUGGINGFACE_HUB_TOKEN"] = "yourtokenhere"
HF_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]

MODELS = [
    {
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "folder": "vae"
    },
    {
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "folder": "text_encoders"
    },
    {
        "repo_id": "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "filename": "Qwen-Image-Edit-2509-Q4_K_M.gguf",
        "folder": "diffusion_models"
    }, 
    {
    "repo_id": "lightx2v/Qwen-Image-Lightning",
    "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
    "folder": "loras"  # For ComfyUI LoRA folder
    }

]

def download_model(repo_id, filename, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Extract just the bare filename (no subdirectory path)
    bare_filename = os.path.basename(filename)
    final_path = os.path.join(folder, bare_filename)

    # Skip if already downloaded
    if os.path.exists(final_path):
        print(f"⏭️  Already exists, skipping: {final_path}\n")
        return

    try:
        print(f"Downloading {filename} from {repo_id} → {folder}/ ...")

        # Download to a temp cache dir to avoid nested path issues
        tmp_dir = os.path.join(folder, "_tmp_cache")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN
        )

        # Move the file to the flat target location
        shutil.move(downloaded_path, final_path)
        print(f"✅ Saved to: {final_path}\n")

        # Clean up the temp cache dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    except HfHubHTTPError as e:
        print(f"❌ Hugging Face error for {filename}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error for {filename}: {e}")

if __name__ == "__main__":
    for model in MODELS:
        download_model(model["repo_id"], model["filename"], model["folder"])
