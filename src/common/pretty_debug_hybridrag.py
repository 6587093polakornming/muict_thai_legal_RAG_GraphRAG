def pretty_print_rag(debug_result:dict):
        """
            แสดงผลลัพธ์ของกระบวนการ RAG ให้สวยงามและอ่านง่าย
            Args:
                debug_result (dict): Dictionary ที่เก็บข้อมูลการ Debug ประกอบด้วย:
                    - query (str): คำถามที่ส่งเข้าไป
                    - num_candidates (int): จำนวน Candidate ที่ค้นหาได้เบื้องต้น
                    - rerank_candidates (int): จำนวนที่เหลือหลังการ Rerank
                    - final (int): จำนวนเอกสารสุดท้ายที่ใช้
                    - docs_candidates (list): รายการเอกสารทั้งหมด
                    - context (str): เนื้อหาที่ถูกนำไปใส่ใน Prompt
                    - answer (str): คำตอบจาก LLM
                    - token (dict): รายละเอียดการใช้ Token
                    - time_elapsed (float): เวลาที่ใช้ในการประมวลผล
        """
        # 3. ดึงตัวแปรต่างๆ ออกมาใช้งาน
        answer = debug_result["answer"]
        docs = debug_result["docs_candidates"]
        tokens = debug_result["token"]
        times = debug_result["time_elapsed"]
        context = debug_result["context"]

        print("\n" + "-" * 30)
        print("⌛ กำลังค้นหาและประมวลผลคำตอบ...")
        print("-" * 30)

        # 4. แสดงผลลัพธ์หลัก (Answer)
        print(f"\n💡 คำตอบ:\n{answer}")

        # 5. แสดงสถิติและเวลา (Latency & Stats)
        print(f"\n⏱️  Latency & Stats:")
        print(f"   - Retrieval Time: {times['retrieve_time']:.3f} s")
        print(f"   - LLM Time:       {times['llm_time']:.3f} s")
        print(f"   - Total Time:     {times['total_elapsed']:.3f} s")
        print(
            f"   - Candidates Found: {debug_result['num_candidates']} -> Final Docs: {debug_result['final']}"
        )

        # 6. แสดงข้อมูลการใช้ Token
        print(f"\n🎫 Token Usage:")
        print(f"   - Prompt:     {tokens.get('prompt_tokens', 0)}")
        print(f"   - Completion: {tokens.get('completion_tokens', 0)}")
        print(f"   - Total:      {tokens.get('total_tokens', 0)}")

        # 7. แสดงรายการกฎหมายที่ใช้อ้างอิง (Sources)
        print(f"\n📜 แหล่งอ้างอิง (Sources):")
        for doc in docs:
            m = doc.metadata
            print(
                f"   [{m['rank']}] Score: {m['score']:.4f} | {m['law_name']} มาตรา {m['section_num']}"
            )

        # 8. แสดง Context ที่ส่งให้ LLM (สำหรับ Debug)
        print(f"\n📝 FULL CONTEXT SENT TO LLM:")
        print("-" * 60)
        # แสดง context ทั้งหมด หรือตัดมาเฉพาะบางส่วนถ้ามันยาวเกินไป
        print(context)
        print("-" * 60)

        print("\n" + "=" * 60)