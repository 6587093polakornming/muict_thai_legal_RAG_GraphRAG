"""
Unit tests for HybridRetriever._link_ref_law

Test strategy:
- Mock qdrant client.scroll เพื่อไม่ต้องต่อ DB จริง
- ใช้ ScoredPoint / Record จาก qdrant_client.http.models โดยตรง
- แต่ละ test case ตรวจ ordering และ score ของผลลัพธ์

$ python -m pytest test_link_ref_law.py -v
"""

import unittest
from unittest.mock import MagicMock
from qdrant_client.http import models


# ---------------------------------------------------------------------------
# Helper: สร้าง ScoredPoint ง่ายๆ
# ---------------------------------------------------------------------------
def make_scored_point(id_, law_name, section_num, score, reference_laws=None):
    return models.ScoredPoint(
        id=id_,
        score=score,
        version=0,
        payload={
            "law_name": law_name,
            "section_num": section_num,
            "reference_laws": reference_laws or [],
        },
        vector=None,
    )


# สร้าง Record (ผลจาก scroll) ง่ายๆ
def make_record(id_, law_name, section_num):
    return models.ScoredPoint(
        id=id_,
        score=0.0,
        version=0,
        payload={
            "law_name": law_name,
            "section_num": section_num,
            "reference_laws": [],
        },
        vector=None,
    )


# ---------------------------------------------------------------------------
# Stub HybridRetriever — ใช้เฉพาะ _link_ref_law ไม่ต้องโหลด model จริง
# ---------------------------------------------------------------------------
from unittest.mock import patch

# Import เฉพาะ class โดย mock dependency หนัก
import sys
sys.modules.setdefault("FlagEmbedding", MagicMock())
sys.modules.setdefault("langchain_core", MagicMock())
sys.modules.setdefault("langchain_core.documents", MagicMock())

# Mock config module
mock_config_module = MagicMock()
sys.modules["retriever"] = MagicMock()
sys.modules["retriever.config"] = mock_config_module


def make_retriever(scroll_return):
    """
    สร้าง HybridRetriever stub พร้อม mock client.scroll
    scroll_return: list ของ Record ที่ต้องการให้ scroll คืนกลับมา
    """
    from src.rag.hybrid_retriever import HybridRetriever

    config = MagicMock()
    config.collection_name = "test_collection"

    client = MagicMock()
    client.scroll.return_value = (scroll_return, None)

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.client = client
    retriever.config = config
    return retriever


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestLinkRefLaw(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Edge Cases
    # ------------------------------------------------------------------

    def test_empty_input_returns_empty(self):
        """list_pts ว่าง → return []"""
        r = make_retriever([])
        result = r._link_ref_law([])
        self.assertEqual(result, [])

    def test_no_reference_laws_returns_original(self):
        """ไม่มี reference_laws ใน payload → return list_pts เดิม"""
        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9)
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)

        r = make_retriever([])
        result = r._link_ref_law([A, B])

        self.assertEqual([p.id for p in result], [1, 2])

    # ------------------------------------------------------------------
    # 2. top-1 + parent-first (default)
    # ------------------------------------------------------------------

    def test_top1_parent_first_ordering(self):
        """
        top-1 + parent-first:
        A มี ref → [ref10, ref20]
        expected: [A, ref10, ref20, B, C]
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref20 = make_record(20, "กฎหมายแพ่ง", "200")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[
                                  {"law_name": "กฎหมายแพ่ง", "section_num": "100"},
                                  {"law_name": "กฎหมายแพ่ง", "section_num": "200"},
                              ])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)
        C = make_scored_point(3, "กฎหมายแพ่ง", "12", score=0.7)

        r = make_retriever([ref10, ref20])
        result = r._link_ref_law([A, B, C], expansion_mode="top-1", reorder_mode="parent-first")

        self.assertEqual([p.id for p in result], [1, 10, 20, 2, 3])

    def test_top1_parent_first_ref_score_equals_parent_score(self):
        """ref law ต้องได้ score เท่ากับ parent (A.score=0.9)"""
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])

        r = make_retriever([ref10])
        result = r._link_ref_law([A], expansion_mode="top-1", reorder_mode="parent-first")

        ref_result = next(p for p in result if p.id == 10)
        self.assertAlmostEqual(ref_result.score, 0.9)

    # ------------------------------------------------------------------
    # 3. top-1 + append-last
    # ------------------------------------------------------------------

    def test_top1_append_last_ordering(self):
        """
        top-1 + append-last:
        expected: [A, B, C, ref10, ref20]
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref20 = make_record(20, "กฎหมายแพ่ง", "200")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[
                                  {"law_name": "กฎหมายแพ่ง", "section_num": "100"},
                                  {"law_name": "กฎหมายแพ่ง", "section_num": "200"},
                              ])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)
        C = make_scored_point(3, "กฎหมายแพ่ง", "12", score=0.7)

        r = make_retriever([ref10, ref20])
        result = r._link_ref_law([A, B, C], expansion_mode="top-1", reorder_mode="append-last")

        self.assertEqual([p.id for p in result], [1, 2, 3, 10, 20])

    def test_top1_append_last_ref_score_equals_parent_score(self):
        """append-last: ref ต้องได้ score ของ parent ที่อ้างถึง"""
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)

        r = make_retriever([ref10])
        result = r._link_ref_law([A, B], expansion_mode="top-1", reorder_mode="append-last")

        ref_result = next(p for p in result if p.id == 10)
        self.assertAlmostEqual(ref_result.score, 0.9)

    # ------------------------------------------------------------------
    # 4. top-n + parent-first
    # ------------------------------------------------------------------

    def test_topn_parent_first_ordering(self):
        """
        top-n + parent-first:
        A.ref=[ref10], B.ref=[ref30]
        expected: [A, ref10, B, ref30, C]

        หมายเหตุ: ถ้า code ยังใช้ batch_map รวม (bug เดิม)
        ref30 จะถูกเพิ่มตอน loop A → [A, ref10, ref30, B, C]  ← ผิด
        test นี้จะ fail และบอกว่ายังมี bug อยู่
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref30 = make_record(30, "กฎหมายแพ่ง", "300")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "300"}])
        C = make_scored_point(3, "กฎหมายแพ่ง", "12", score=0.7)

        r = make_retriever([ref10, ref30])
        result = r._link_ref_law([A, B, C], expansion_mode="top-n", reorder_mode="parent-first")

        self.assertEqual([p.id for p in result], [1, 10, 2, 30, 3])

    def test_topn_parent_first_ref_score_matches_own_parent(self):
        """
        top-n + parent-first: ref แต่ละตัวต้องได้ score ของ parent ตัวเอง
        ref10 → score ของ A (0.9)
        ref30 → score ของ B (0.8)
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref30 = make_record(30, "กฎหมายแพ่ง", "300")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "300"}])

        r = make_retriever([ref10, ref30])
        result = r._link_ref_law([A, B], expansion_mode="top-n", reorder_mode="parent-first")

        scores = {p.id: p.score for p in result}
        self.assertAlmostEqual(scores[10], 0.9)  # ref10 ต้องได้ score A
        self.assertAlmostEqual(scores[30], 0.8)  # ref30 ต้องได้ score B

    # ------------------------------------------------------------------
    # 5. top-n + append-last
    # ------------------------------------------------------------------

    def test_topn_append_last_ordering(self):
        """
        top-n + append-last:
        A.ref=[ref10], B.ref=[ref30]
        expected: [A, B, C, ref10, ref30]
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref30 = make_record(30, "กฎหมายแพ่ง", "300")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "300"}])
        C = make_scored_point(3, "กฎหมายแพ่ง", "12", score=0.7)

        r = make_retriever([ref10, ref30])
        result = r._link_ref_law([A, B, C], expansion_mode="top-n", reorder_mode="append-last")

        self.assertEqual([p.id for p in result], [1, 2, 3, 10, 30])

    def test_topn_append_last_ref_score_matches_own_parent(self):
        """
        top-n + append-last: ref แต่ละตัวต้องได้ score ของ parent ตัวเอง
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")
        ref30 = make_record(30, "กฎหมายแพ่ง", "300")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "300"}])

        r = make_retriever([ref10, ref30])
        result = r._link_ref_law([A, B], expansion_mode="top-n", reorder_mode="append-last")

        scores = {p.id: p.score for p in result}
        self.assertAlmostEqual(scores[10], 0.9)
        self.assertAlmostEqual(scores[30], 0.8)

    # ------------------------------------------------------------------
    # 6. Deduplication
    # ------------------------------------------------------------------

    def test_no_duplicate_in_results(self):
        """ไม่มี id ซ้ำในผลลัพธ์ไม่ว่า mode ไหน"""
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])

        r = make_retriever([ref10])
        for mode in [("top-n", "parent-first"), ("top-n", "append-last")]:
            with self.subTest(mode=mode):
                result = r._link_ref_law([A, B], expansion_mode=mode[0], reorder_mode=mode[1])
                ids = [p.id for p in result]
                self.assertEqual(len(ids), len(set(ids)), f"พบ duplicate ids: {ids}")

    def test_top1_only_expands_first_context(self):
        """
        top-1: B มี ref แต่ไม่ถูก expand เพราะ expansion_mode=top-1
        scroll จะถูกเรียกด้วย ref ของ A เท่านั้น
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                              reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "999"}])

        r = make_retriever([ref10])
        result = r._link_ref_law([A, B], expansion_mode="top-1", reorder_mode="parent-first")

        ids = [p.id for p in result]
        # ref ของ B (999) ไม่ควรอยู่ใน result
        self.assertIn(10, ids)
        self.assertNotIn(999, ids)

    # ------------------------------------------------------------------
    # 7. ref law ที่อยู่ใน list_pts อยู่แล้ว
    # ------------------------------------------------------------------

    def test_ref_already_in_list_pts_not_duplicated_parent_first(self):
        """
        ref10 อยู่ใน list_pts อยู่แล้ว (ตำแหน่ง 2)
        top-1 + parent-first: ต้องไม่ถูกเพิ่มซ้ำ
        expected: [A, ref10, B]  — ref10 คงอยู่ที่เดิม ไม่ถูกแทรกซ้ำหลัง A
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                            reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        ref10_in_list = make_scored_point(10, "กฎหมายแพ่ง", "100", score=0.75)
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)

        r = make_retriever([ref10])
        result = r._link_ref_law([A, ref10_in_list, B],
                                expansion_mode="top-1", reorder_mode="parent-first")

        ids = [p.id for p in result]
        self.assertEqual(ids.count(10), 1, "ref10 ต้องไม่ซ้ำ")

    def test_ref_already_in_list_pts_not_duplicated_append_last(self):
        """
        ref10 อยู่ใน list_pts อยู่แล้ว
        top-1 + append-last: ต้องไม่ถูกเพิ่มซ้ำต่อท้าย
        expected: [A, ref10_in_list, B]  ไม่มี ref10 ต่อท้าย
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                            reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        ref10_in_list = make_scored_point(10, "กฎหมายแพ่ง", "100", score=0.75)
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)

        r = make_retriever([ref10])
        result = r._link_ref_law([A, ref10_in_list, B],
                                expansion_mode="top-1", reorder_mode="append-last")

        ids = [p.id for p in result]
        self.assertEqual(ids.count(10), 1, "ref10 ต้องไม่ซ้ำ")
        self.assertEqual(ids, [1, 10, 2], "ordering ต้องคงเดิม")

    # ------------------------------------------------------------------
    # 8. ref ซ้ำข้าม parent (top-n) — parent แรกได้ score
    # ------------------------------------------------------------------

    def test_shared_ref_across_parents_score_from_first_parent(self):
        """
        A.ref=[ref10], B.ref=[ref10]  ← ref เดียวกัน
        ref10 ควรได้ score ของ A (parent แรกที่เจอ)
        และปรากฏแค่ครั้งเดียว
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                            reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8,
                            reference_laws=[{"law_name": "กฎหมายแพ่ง", "section_num": "100"}])

        r = make_retriever([ref10])
        for reorder in ["parent-first", "append-last"]:
            with self.subTest(reorder=reorder):
                result = r._link_ref_law([A, B], expansion_mode="top-n", reorder_mode=reorder)
                ids = [p.id for p in result]
                self.assertEqual(ids.count(10), 1, "ref10 ต้องปรากฏแค่ครั้งเดียว")
                scores = {p.id: p.score for p in result}
                self.assertAlmostEqual(scores[10], 0.9, msg="ref10 ต้องได้ score ของ A (parent แรก)")

    # ------------------------------------------------------------------
    # 9. scroll คืน record ไม่ครบ (ref missing จาก DB)
    # ------------------------------------------------------------------

    def test_missing_ref_from_db_does_not_crash(self):
        """
        A.ref=[ref10, ref_missing]
        scroll คืนแค่ [ref10] — ref_missing ไม่มีใน DB
        ต้องไม่ crash และ result มีแค่ ref10
        """
        ref10 = make_record(10, "กฎหมายแพ่ง", "100")

        A = make_scored_point(1, "กฎหมายแพ่ง", "10", score=0.9,
                            reference_laws=[
                                {"law_name": "กฎหมายแพ่ง", "section_num": "100"},
                                {"law_name": "กฎหมายแพ่ง", "section_num": "999"},  # ไม่มีใน DB
                            ])
        B = make_scored_point(2, "กฎหมายแพ่ง", "11", score=0.8)

        r = make_retriever([ref10])  # scroll คืนแค่ ref10
        for reorder in ["parent-first", "append-last"]:
            with self.subTest(reorder=reorder):
                result = r._link_ref_law([A, B], expansion_mode="top-1", reorder_mode=reorder)
                ids = [p.id for p in result]
                self.assertIn(10, ids,  "ref10 ต้องอยู่ใน result")
                self.assertNotIn(999, ids, "ref_missing ต้องไม่อยู่ใน result")


if __name__ == "__main__":
    unittest.main(verbosity=2)