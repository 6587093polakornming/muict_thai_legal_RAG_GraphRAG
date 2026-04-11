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

    print(f"\n💡 คำตอบ:\n{answer}")

    print(f"\n⏱️ Latency & Stats:")
    print(f"   - Retrieval Time: {times['retrieve_time']:.3f} s")
    print(f"   - LLM Time:       {times['llm_time']:.3f} s")
    print(f"   - Total Time:     {times['total_elapsed']:.3f} s")
    print(
        f"   - Final Docs: {debug_result['final']}"
    )

    print(f"\n🎫 Token Usage:")
    print(f"   - Prompt:      {tokens.get('prompt_tokens', 0)}")
    print(f"   - Completion:  {tokens.get('completion_tokens', 0)}")
    print(f"   - Total:       {tokens.get('total_tokens', 0)}")

    print(f"\n📜 แหล่งอ้างอิง (Sources):")
    
    for i, doc in enumerate(docs, 1):
        m = doc.metadata if hasattr(doc, 'metadata') else {}

        # FIX: score is already stored in metadata from get_retrieve()
        score = float(m.get('score', 0.0))
        law   = m.get('law_name', 'ไม่ระบุชื่อกฎหมาย')
        sec   = m.get('section_num', 'N/A')

        print(f"   [{i}] Score: {score:.4f} | {law} มาตรา {sec}")

        children = m.get('reference_laws', [])  # FIX: key is 'reference_laws', not 'children'
        if children:
            for child in children:
                c_law = child.get('law_name', '')
                c_sec = child.get('section_num', '')
                print(f"       ↳ อ้างถึง: {c_law} มาตรา {c_sec}")

    print(f"\n📝 FULL CONTEXT SENT TO LLM:")
    print("-" * 60)
    print(context)
    print("-" * 60)
    print("\n" + "=" * 60)