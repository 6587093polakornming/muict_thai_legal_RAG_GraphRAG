import os
from pathlib import Path

def create_project_structure():
    # กำหนดโครงสร้าง Folder ที่ต้องการ (Relative Paths)
    structure = [
        "data/raw",
        "data/processed",
        "data/tests",
        "docker",
        "notebooks",
        "src/common",
        "src/rag",
        "src/graph_rag",
        "src/evaluation",
    ]

    print("🚀 Starting project folder setup...")

    for folder in structure:
        path = Path(folder)
        
        # ตรวจสอบก่อนสร้าง: ถ้ายังไม่มี Folder นี้อยู่ ให้สร้างใหม่
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created folder: {folder}")
        else:
            # ถ้ามีอยู่แล้ว แจ้งเตือนแต่ไม่ทำอะไรเพิ่ม (ป้องกันการทับซ้อน)
            print(f"🟡 Folder already exists: {folder}")

    print("\n✨ All folders are set up. Ready for development!")

if __name__ == "__main__":
    create_project_structure()