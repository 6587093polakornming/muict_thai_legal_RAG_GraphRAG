import pandas as pd
import torch
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict
import json
from tqdm import tqdm

class EmbeddingPipeline:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BGEM3FlagModel(
            model_name, 
            use_fp16=(self.device == 'cuda'), 
            device=self.device
        )
        print(f"🚀 Model loaded on: {self.device.upper()}")

    def process_dataframe(self, df: pd.DataFrame, text_col: str, output_file: Path, batch_size: int = 16):
        # เปิดไฟล์รอไว้เลย (โหมด Append)
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(df), batch_size), desc="Embedding & Saving"):
                batch_df = df.iloc[i : i + batch_size]
                texts = batch_df[text_col].tolist()
                
                outputs = self.model.encode(texts, return_dense=True, return_sparse=True)
                
                for row_tuple, dense, sparse in zip(batch_df.itertuples(index=True), 
                                                 outputs['dense_vecs'], 
                                                 outputs['lexical_weights']):
                    clean_sparse = {str(k): float(v) for k, v in sparse.items() if v is not None}
                    
                    record = {
                        'id': row_tuple[0],
                        'law_name': row_tuple.law_name,
                        'section_num': row_tuple.section_num,
                        'text': getattr(row_tuple, text_col),
                        'dense_vector': dense.tolist(),
                        'sparse_vector': json.dumps(clean_sparse)
                    }
                    # เขียนลงไฟล์ทีละแถว (JSON Lines)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    # --- 1. Configuration ---
    BASE_DIR = Path.cwd()
    INPUT_FILE = BASE_DIR / "data" / "processed" / "nitibench_cleaned_2026-03-17.parquet"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "vectors_sparse_nitibench-ccl-bge-m3.parquet"
    MODEL_NAME = "VISAI-AI/nitibench-ccl-human-finetuned-bge-m3"
    # ... Config ...
    TEMP_JSONL = OUTPUT_PATH.with_suffix('.jsonl') # ไฟล์ชั่วคราว

    # --- 2. Initialize ---
    pipeline = EmbeddingPipeline(MODEL_NAME)

    # ตรวจสอบไฟล์ก่อนโหลด (Best Practice)
    if not INPUT_FILE.exists():
        print(f"❌ Error: File not found at {INPUT_FILE}")
        return
    df = pd.read_parquet(INPUT_FILE)

    # รัน process (เขียนลงไฟล์ไปเรื่อยๆ)
    pipeline.process_dataframe(df, text_col='section_content', output_file=TEMP_JSONL, batch_size=16)
    
    # สุดท้าย: โหลด JSONL กลับมาเซฟเป็น Parquet ทีเดียว (ขั้นตอนนี้จะเร็ว)
    print("📦 Converting JSONL to Parquet...")
    df_final = pd.read_json(TEMP_JSONL, lines=True)
    df_final.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
    # ลบไฟล์ชั่วคราวทิ้ง
    if TEMP_JSONL.exists():
        TEMP_JSONL.unlink()

    print(f"✅ Finished! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()