import subprocess
import os
from pathlib import Path

def clone_datasets():
    # รายชื่อ repository ที่ต้องการ clone
    datasets = [
        "https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG",
        "https://huggingface.co/datasets/VISAI-AI/nitibench"
    ]
    
    # กำหนดเป้าหมายไปที่ data/raw ตามโครงสร้างที่คุณวางไว้
    target_dir = Path("data/raw")
    
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Target directory: {target_dir.absolute()}")

    for url in datasets:
        # ดึงชื่อโฟลเดอร์จาก URL (เช่น nitibench)
        repo_name = url.split("/")[-1]
        repo_path = target_dir / repo_name
        
        if repo_path.exists():
            print(f"🟡 Skip: '{repo_name}' already exists in data/raw")
            continue
            
        print(f"🚀 Cloning {repo_name}...")
        try:
            # รันคำสั่ง git clone โดยกำหนด cwd (current working directory) ไปที่ data/raw
            subprocess.run(["git", "clone", url], cwd=target_dir, check=True)
            print(f"✅ Successfully cloned: {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning {repo_name}: {e}")
        except FileNotFoundError:
            print("❌ Error: 'git' command not found. Please install Git first.")

if __name__ == "__main__":
    clone_datasets()