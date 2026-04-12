import json
import argparse
from pathlib import Path

def split_jsonl_by_ref(input_path, output_has_ref, output_no_ref):
    """
    แยกไฟล์ JSONL ออกเป็น 2 ไฟล์ตามเงื่อนไขการมี Reference Laws
    
    Args:
        input_path (str): เส้นทางไปยังไฟล์ .jsonl ต้นทาง (เช่น 'nitibench.jsonl')
        output_has_ref (str): ชื่อไฟล์สำหรับเก็บข้อมูลที่มีรายการอ้างอิงกฎหมาย
        output_no_ref (str): ชื่อไฟล์สำหรับเก็บข้อมูลที่ไม่มีการอ้างอิงกฎหมาย
        
    Logic:
        เช็คฟิลด์ 'reference_laws' ในแต่ละ Record:
        - ถ้ามีข้อมูลใน List (> 0) จะถูกเขียนลงไฟล์ output_has_ref
        - ถ้าเป็น List ว่างหรือไม่มีฟิลด์นี้ จะถูกเขียนลงไฟล์ output_no_ref
    """
    input_file = Path(input_path)
    
    # ตรวจสอบความปลอดภัยของไฟล์ต้นทาง
    if not input_file.exists():
        print(f"❌ Error: ไม่พบไฟล์ที่ตำแหน่ง {input_path}")
        return

    try:
        # เปิดไฟล์ Input และ Output พร้อมกัน (ใช้ encoding utf-8 สำหรับภาษาไทย)
        with input_file.open('r', encoding='utf-8') as f_in, \
             Path(output_has_ref).open('w', encoding='utf-8') as f_out_has, \
             Path(output_no_ref).open('w', encoding='utf-8') as f_out_no:
            
            count_has = 0
            count_no = 0

            # วนลูปอ่านข้อมูลทีละบรรทัด (ประหยัด Memory)
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    record = json.loads(line)
                    
                    # ตรวจสอบฟิลด์ reference_laws
                    # 1. ต้องมีฟิลด์นี้อยู่ 2. ต้องเป็น list 3. ต้องมีข้อมูลข้างใน
                    refs = record.get('gt_reference_laws')
                    if isinstance(refs, list) and len(refs) > 0:
                        f_out_has.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count_has += 1
                    else:
                        f_out_no.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count_no += 1
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: บรรทัดที่ {line_num} รูปแบบ JSON ไม่ถูกต้อง (ข้ามการทำงาน)")

        # สรุปผลการทำงานออกมาทางหน้าจอ
        print(f"\n{'='*30}")
        print(f"✅ ประมวลผลสำเร็จ!")
        print(f"{'='*30}")
        print(f"📝 รวมทั้งหมด: {count_has + count_no} รายการ")
        print(f"📂 มี Reference ({count_has} ข้อ): {output_has_ref}")
        print(f"📂 ไม่มี Reference ({count_no} ข้อ): {output_no_ref}")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")

if __name__ == "__main__":
    # ส่วนจัดการ Command Line Arguments
    parser = argparse.ArgumentParser(
        description="เครื่องมือแยก Dataset NitiBench-CCL ตามเงื่อนไขการมี Reference Laws",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # กำหนด Parameter หลัก
    parser.add_argument("input_file", help="Path ของไฟล์ .jsonl ที่ต้องการนำมาแยก")

    # ขั้นตอนการตั้งชื่อไฟล์ Output อัตโนมัติ (Default Naming)
    args_pre, _ = parser.parse_known_args()
    if args_pre.input_file:
        input_path = Path(args_pre.input_file)
        # ตัวอย่าง: data.jsonl -> data_has_ref.jsonl
        default_has = str(input_path.with_name(f"{input_path.stem}_has_ref.jsonl"))
        default_no = str(input_path.with_name(f"{input_path.stem}_no_ref.jsonl"))
    else:
        default_has = "has_ref.jsonl"
        default_no = "no_ref.jsonl"

    # ตัวเลือกเสริม (Optional Parameters)
    parser.add_argument("--output_has_ref", default=default_has, help=f"ไฟล์ปลายทางสำหรับข้อมูลที่มี Ref\n(Default: {default_has})")
    parser.add_argument("--output_no_ref", default=default_no, help=f"ไฟล์ปลายทางสำหรับข้อมูลที่ไม่มี Ref\n(Default: {default_no})")

    args = parser.parse_args()

    # เริ่มการทำงาน
    split_jsonl_by_ref(args.input_file, args.output_has_ref, args.output_no_ref)

# ==========================================
# วิธีใช้งาน (How to use)
# ==========================================
# 1. การใช้งานแบบพื้นฐาน (ใช้ชื่อไฟล์อัตโนมัติ):
#    python split_data.py data/evaluation/results_hybrid.jsonl
#
# 2. การระบุชื่อไฟล์ Output เอง:
#    python split_data.py input.jsonl --output_has_ref rich_context.jsonl --output_no_ref plain.jsonl
#
# 3. ตรวจสอบวิธีใช้ผ่าน Help:
#    python split_data.py --help
# ==========================================