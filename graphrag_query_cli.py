import time
import pandas as pd
from src.graph_rag.graph_builder import LegalKGManager
from src.graph_rag.graphrag_retriever import LegalRetriever
from src.common.pretty_debug_graphrag import pretty_print_rag
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimates token count using tiktoken."""
    try:
        # Use the encoding for the specific model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to the standard encoding used by most modern models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

if __name__ == "__main__":
    # --- 1. Ingestion (Run once to build the graph) ---
    # manager = LegalKGManager()
    # df = pd.read_parquet('path_to_your_data.parquet')
    # print("Starting Ingestion...")
    # manager.import_legal_data(df)
    # manager.close()
    
    # --- 2. Querying ---
    legal_rag = LegalRetriever()
    cypher_query = """
            OPTIONAL MATCH (node)-[:REFERENCES_TO]->(referencedSection)
            WITH node, score, 
                collect({
                    law_name: referencedSection.law_name, 
                    section_num: referencedSection.section_num,
                    text: referencedSection.text
                }) AS children_nodes
            RETURN 
                node.text AS content, 
                score, 
                {
                    parent_law_name: node.law_name,
                    parent_section_num: node.section_num,
                    children: [c in children_nodes WHERE c.law_name IS NOT NULL]
                } AS metadata
            """

    
    while True:
        query = input("\nEnter your legal question (or type 'exit' to quit): ")
        p_tokens = count_tokens(f"{cypher_query}\n\nQuestion: {query}")
        if query.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            legal_rag.driver.close()
            break
        # query = "ถ้ามีคนประกอบกิจการในลักษณะเป็นศูนย์ซื้อขายสัญญาซื้อขายล่วงหน้าโดยไม่ได้รับใบอนุญาตต้องระวางโทษอย่างไร"

        # Tracking time for debug info
        start_total = time.perf_counter()
        
        # Simulating the structured output your pretty_print function expects
        # In a real scenario, you'd wrap the retriever logic to return this dict
        try:
            # Retrieve context
            start_retrieval = time.perf_counter()
            retriever = legal_rag.get_retriever(cypher_query=cypher_query)
            retrieved_docs = retriever.search(query_text=query, top_k=5)
            retrieve_time = time.perf_counter() - start_retrieval
            
            # Generate Answer
            start_llm = time.perf_counter()
            rag_response = legal_rag.search(query)
            llm_time = time.perf_counter() - start_llm
            
            total_elapsed = time.perf_counter() - start_total
            c_tokens = count_tokens(rag_response.answer)
            
            # Calculate prompt tokens from retrieved docs (simplified estimation)
            for i in range(len(retrieved_docs.items)):
                p_tokens += count_tokens(retrieved_docs.items[i].content)

            # Mocking the debug_result dictionary format requested
            debug_result = {
                "query": query,
                "num_candidates": len(retrieved_docs.items),
                "final": len(retrieved_docs.items),
                "docs_candidates": retrieved_docs.items,
                "context": "\n".join([item.content for item in retrieved_docs.items]),
                "answer": rag_response.answer,
                "token": {
                    "prompt_tokens": p_tokens, # Note: Actual tokens require callback listeners
                    "completion_tokens": c_tokens,
                    "total_tokens": p_tokens + c_tokens
                },
                "time_elapsed": {
                    "retrieve_time": retrieve_time,
                    "llm_time": llm_time,
                    "total_elapsed": total_elapsed
                }
            }

            # --- 3. Display Result ---
            pretty_print_rag(debug_result)

        except Exception as e:
            print(f"An error occurred: {e}")

