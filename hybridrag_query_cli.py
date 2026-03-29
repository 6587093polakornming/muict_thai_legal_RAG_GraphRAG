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
# FINAL_LIMIT = 3  # top-k returned to LLM
RERANK_LIMIT = 3


if __name__ == "__main__":
    # --- Configure your LLM here ---
    llm = ChatOpenAI(
         model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
        # model_name="openai/gpt-4o-mini",  # หรือรุ่นที่ท่านต้องการใช้
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
        # openai_api_base="https://openrouter.ai/api/v1",  # สำคัญ: ใส่แทน base_url เดิม
        temperature=0,
        max_tokens=4096,
    )

    # --- Build RAG ---
    rag = ThaiLegalRAG(llm=llm, retrieval_limit=RETRIEVAL_LIMIT, reranking_limit=RERANK_LIMIT)

    # # --- Simple chat ---
    # question = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
    # answer = rag.chat(question)
    # print(answer)

    ### Create Loop input user and Counting Time
    while True:
        # 1. รับ input จาก user
        question = input("\nคำถามของคุณ (พิมพ์ 'exit' เพื่อจบการทำงาน): ").strip()

        # เช็คเงื่อนไขเพื่อออกจาก Loop
        if question.lower() in ["exit", "quit", "ออก"]:
            print("ปิดระบบ... สวัสดีครับ")
            break

        if not question:
            continue

        print("\n" + "-"*30)
        print("⌛ กำลังค้นหาและประมวลผลคำตอบ...")
        print("-"*30)

        # 2. เรียกใช้งาน RAG ผ่าน debug function
        debug_result = rag.debug(question)

        # 3. ดึงตัวแปรต่างๆ ออกมาใช้งาน
        answer = debug_result["answer"]
        docs = debug_result["docs_candidates"]
        tokens = debug_result["token"]
        times = debug_result["time_elapsed"]
        context = debug_result["context"]

        # 4. แสดงผลลัพธ์หลัก (Answer)
        print(f"\n💡 คำตอบ:\n{answer}")

        # 5. แสดงสถิติและเวลา (Latency & Stats)
        print(f"\n⏱️  Latency & Stats:")
        print(f"   - Retrieval Time: {times['retrieve_time']:.3f} s")
        print(f"   - LLM Time:       {times['llm_time']:.3f} s")
        print(f"   - Total Time:     {times['total_elapsed']:.3f} s")
        print(f"   - Candidates Found: {debug_result['num_candidates']} -> Final Docs: {debug_result['final']}")

        # 6. แสดงข้อมูลการใช้ Token
        print(f"\n🎫 Token Usage:")
        print(f"   - Prompt:     {tokens.get('prompt_tokens', 0)}")
        print(f"   - Completion: {tokens.get('completion_tokens', 0)}")
        print(f"   - Total:      {tokens.get('total_tokens', 0)}")

        # 7. แสดงรายการกฎหมายที่ใช้อ้างอิง (Sources)
        print(f"\n📜 แหล่งอ้างอิง (Sources):")
        for doc in docs:
            m = doc.metadata
            print(f"   [{m['rank']}] Score: {m['score']:.4f} | {m['law_name']} มาตรา {m['section_num']}")

        # 8. แสดง Context ที่ส่งให้ LLM (สำหรับ Debug)
        print(f"\n📝 FULL CONTEXT SENT TO LLM:")
        print("-" * 60)
        # แสดง context ทั้งหมด หรือตัดมาเฉพาะบางส่วนถ้ามันยาวเกินไป
        print(context) 
        print("-" * 60)

        print("\n" + "="*60)
        