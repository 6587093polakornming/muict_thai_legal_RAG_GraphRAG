from langchain_openai import ChatOpenAI
import os
from src.graph_rag.graphrag_retriever import GraphRAGRetriever
from src.common.pretty_debug_graphrag import pretty_print_rag


if __name__ == "__main__":
    # --- 1. Ingestion (Run once to build the graph) ---
    # manager = LegalKGManager()
    # df = pd.read_parquet('path_to_your_data.parquet')
    # print("Starting Ingestion...")
    # manager.import_legal_data(df)
    # manager.close()
    
    # --- 2. Querying ---
    llm= ChatOpenAI(
            model_name="typhoon-v2.5-30b-a3b-instruct",  # หรือรุ่นที่ท่านต้องการใช้
            # model_name="openai/gpt-4o-mini",  # หรือรุ่นที่ท่านต้องการใช้
            openai_api_key=os.getenv("thai_llm_API_key"),
            openai_api_base="https://api.opentyphoon.ai/v1",  # สำคัญ: ใส่แทน base_url เดิม
            # openai_api_base="https://openrouter.ai/api/v1",  # สำคัญ: ใส่แทน base_url เดิม
            temperature=0,
            max_tokens=16384,
    )
    legal_rag = GraphRAGRetriever(llm=llm)

    
    while True:
        query = input("\nEnter your legal question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            legal_rag.driver.close()
            break
        # query = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"
        try:
            debug_result = legal_rag.debug(query=query)

            # --- 3. Display Result ---
            pretty_print_rag(debug_result)

        except Exception as e:
            print(f"An error occurred: {e}")

