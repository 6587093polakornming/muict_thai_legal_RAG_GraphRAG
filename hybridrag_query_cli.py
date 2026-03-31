from __future__ import annotations
import time
import os

# from langchain_openai import ChatOpenAI  # swap to langchain_anthropic, langchain_google_genai, etc.
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from src.common.pretty_debug_hybridrag import pretty_print_rag
from src.rag.config import RAGConfig
from src.rag.hybridrag_langhchain import ThaiLegalRAG

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    # --- Configure your LLM here ---
    llm = ChatOpenAI(
        model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
        # model_name="openai/gpt-4o-mini",  # หรือรุ่นที่ท่านต้องการใช้
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
        # openai_api_base="https://openrouter.ai/api/v1",  # สำคัญ: ใส่แทน base_url เดิม
        temperature=0,
        max_tokens=8192,
    )

    # --- Build RAG ---
    config = RAGConfig(
        retrieval_limit=3,
        reranking_limit=3,
    )
    rag = ThaiLegalRAG(llm=llm, config=config)

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
        # 2. เรียกใช้งาน RAG ผ่าน debug function

        debug_result = rag.debug(question)
        pretty_print_rag(debug_result)
