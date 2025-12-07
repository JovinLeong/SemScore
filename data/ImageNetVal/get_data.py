from huggingface_hub import hf_hub_download,login
import shutil
import os
import zipfile

token = "your_huggingface_token"
save_path = "."  
extract_dir = "./"

# Replace with your huggingface token
login(token=token)

# Download to cache 
cached_path = hf_hub_download(
    repo_id="mlx-vision/imagenet-1k",
    filename="val.zip",
    repo_type="dataset",
)

# Copy to dir
os.makedirs(os.path.dirname(save_path), exist_ok=True)
shutil.copy(cached_path, save_path)

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile('val.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
