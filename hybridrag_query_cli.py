from __future__ import annotations
import time
import os

# from langchain_openai import ChatOpenAI  # swap to langchain_anthropic, langchain_google_genai, etc.
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from src.rag.hybridrag_langhchain import ThaiLegalRAG

from dotenv import load_dotenv
load_dotenv()  # โหลดค่าจาก .env

RETRIEVAL_LIMIT = 3  # candidates sent to reranker
FINAL_LIMIT = 3  # top-k returned to LLM


if __name__ == "__main__":
    # --- Configure your LLM here ---
    llm = ChatOpenAI(
        model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
        temperature=0,
        max_tokens=4096,
    )

    # --- Build RAG ---
    rag = ThaiLegalRAG(llm=llm, retrieval_limit=RETRIEVAL_LIMIT, final_limit=FINAL_LIMIT)

    # # --- Simple chat ---
    # question = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
    # answer = rag.chat(question)
    # print(answer)

    # # --- Chat with sources ---
    # answer, sources = rag.chat_with_sources(question)
    # print("\n=== Sources ===")
    # for doc in sources:
    #     m = doc.metadata
    #     print(f"  [{m['law_name']} มาตรา {m['section_num']}] score={m['score']:.4f}")

    ### Create Loop input user and Conuting Time
    while True:
        # 1. รับ input จาก user
        question = input("\nคำถามของคุณ: ").strip()

        # เช็คเงื่อนไขเพื่อออกจาก Loop
        if question.lower() in ["exit", "quit", "ออก"]:
            print("ปิดระบบ... สวัสดีครับ")
            break

        if not question:
            continue

        # 2. เริ่มจับเวลา
        print("กำลังค้นหาและประมวลผลคำตอบ...")
        start_time = time.time()

        # 3. เรียกใช้งาน RAG (ใช้ chat_with_sources เพื่อให้เห็นคะแนนด้วย)
        answer, sources = rag.chat_with_sources(question)

        # 4. คำนวณเวลาที่ใช้
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 5. แสดงผลลัพธ์
        print("\n=== คำตอบจาก AI ===")
        print(answer)

        print("\n=== อ้างอิงจากกฎหมาย ===")
        for doc in sources:
            print(f"Score: {doc.metadata["score"]:.4f} | {doc.metadata["law_name"]} มาตรา {doc.metadata["section_num"]}")
            print(doc.page_content)
            

        print(f"\n⏱️ ใช้เวลาประมวลผลทั้งสิ้น: {elapsed_time:.2f} วินาที")
        print("-" * 50)
