import re

def pretty_print_rag(debug_result: dict):
    """
    แสดงผลลัพธ์ของกระบวนการ RAG ให้สวยงามและอ่านง่าย
    """
    answer = debug_result["answer"]
    docs = debug_result["docs_candidates"]
    tokens = debug_result["token"]
    times = debug_result["time_elapsed"]
    context = debug_result["context"]

    print("\n" + "-" * 30)
    print("⌛ กำลังค้นหาและประมวลผลคำตอบ...")
    print("-" * 30)

    # แสดงผลลัพธ์หลัก (Answer)
    print(f"\n💡 คำตอบ:\n{answer}")

    # แสดงสถิติและเวลา (Latency & Stats)
    print(f"\n⏱️ Latency & Stats:")
    print(f"   - Retrieval Time: {times['retrieve_time']:.3f} s")
    print(f"   - LLM Time:       {times['llm_time']:.3f} s")
    print(f"   - Total Time:     {times['total_elapsed']:.3f} s")
    print(
        f"   - Candidates Found: {debug_result['num_candidates']} -> Final Docs: {debug_result['final']}"
    )

    # แสดงข้อมูลการใช้ Token
    print(f"\n🎫 Token Usage:")
    print(f"   - Prompt:      {tokens.get('prompt_tokens', 0)}")
    print(f"   - Completion:  {tokens.get('completion_tokens', 0)}")
    print(f"   - Total:       {tokens.get('total_tokens', 0)}")

    # 7. แสดงรายการกฎหมายที่ใช้อ้างอิง (Sources)
    print(f"\n📜 แหล่งอ้างอิง (Sources):")
    
    docs = debug_result["docs_candidates"]
    
    # We use enumerate to create the [Rank] 1, 2, 3...
    for i, doc in enumerate(docs, 1):
        # Accessing top-level score from the Neo4j Record/Item
        score_match = re.search(r"score=([\d.]+)", str(doc))
        score = float(score_match.group(1)) if score_match else 0.0

        # Accessing metadata dictionary
        m = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Mapping to your specific Neo4j metadata keys
        law = m.get('parent_law_name', 'ไม่ระบุชื่อกฎหมาย')
        sec = m.get('parent_section_num', 'N/A')
        
        print(f"   [{i}] Score: {score:.4f} | {law} มาตรา {sec}")
        
        # Optional: If you want to show the 'Children' (referenced laws)
        children = m.get('children', [])
        if children:
            for child in children:
                c_law = child.get('law_name', '')
                c_sec = child.get('section_num', '')
                print(f"       ↳ อ้างถึง: {c_law} มาตรา {c_sec}")

    # แสดง Context ที่ส่งให้ LLM
    print(f"\n📝 FULL CONTEXT SENT TO LLM:")
    print("-" * 60)
    print(context)
    print("-" * 60)
    print("\n" + "=" * 60)
