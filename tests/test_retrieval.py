"""Unit tests for Arabic RAG Retrieval engine and Policy Knowledge Base."""

import pytest
from src.retrieval import get_knowledge_base, normalize_arabic


@pytest.fixture(scope="module")
def kb():
    """Knowledge base fixture."""
    return get_knowledge_base()


def test_knowledge_base_loaded_all_policies(kb):
    """Verify that all 6 fictional policies are loaded and indexed."""
    assert len(kb.passages) >= 6
    source_ids = {p["source_id"] for p in kb.passages}
    expected_ids = {
        "HR-LEAVE-001",
        "HR-SICK-002",
        "HR-ATTEND-003",
        "HR-REMOTE-004",
        "HR-APPROVAL-005",
        "HR-ABSENCE-006",
    }
    assert expected_ids.issubset(source_ids)


def test_arabic_normalization():
    """Verify normalization of Arabic text (Alef, Taa Marbuta, Diacritics)."""
    raw_text = "إِجَازَةٌ سَنَوِيَّةٌ وَأَيَّامٌ مَطْلُوبَةٌ!"
    normalized = normalize_arabic(raw_text)
    # Diacritics must be removed
    assert "َ" not in normalized
    assert "ٌ" not in normalized
    # Alef must be normalized to 'ا'
    assert "إ" not in normalized
    assert "أ" not in normalized
    # Taa Marbuta at word end normalized to 'ه'
    assert "ة" not in normalized


def test_retrieve_annual_leave_policy(kb):
    """Verify query about annual leave retrieves HR-LEAVE-001."""
    query = "كم رصيد الإجازة السنوية وكيف يتم اقتطاع الأيام؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-LEAVE-001"
    assert len(res.passages) <= 3


def test_retrieve_sick_leave_policy(kb):
    """Verify query about sick leave retrieves HR-SICK-002."""
    query = "ما هي شروط الإجازة المرضية والتقرير الطبي المطلوب؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-SICK-002"


def test_retrieve_attendance_policy(kb):
    """Verify query about working hours and attendance retrieves HR-ATTEND-003."""
    query = "ما هي مواعيد وساعات العمل الرسمية وفترة السماح للحضور؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-ATTEND-003"


def test_retrieve_remote_work_policy(kb):
    """Verify query about remote work retrieves HR-REMOTE-004."""
    query = "هل أستطيع العمل عن بعد وكم يوماً في الأسبوع كحد أقصى؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-REMOTE-004"


def test_retrieve_leave_approval_workflow(kb):
    """Verify query about leave approval workflow retrieves HR-APPROVAL-005."""
    query = "ما هي خطوات مسار اعتماد الإجازة وهل يعتمدها المساعد الذكي؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-APPROVAL-005"


def test_retrieve_absence_deduction_policy(kb):
    """Verify query about absence deduction formula retrieves HR-ABSENCE-006."""
    query = "ما هي معادلة احتساب خصم الغياب غير المدفوع من الراتب؟"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is True
    assert res.top_source_id == "HR-ABSENCE-006"


def test_insufficient_evidence_for_gibberish(kb):
    """Verify evidence gating rejects completely unrelated or gibberish queries."""
    query = "xyz123 qwerty meaningless non-arabic text 9999"
    res = kb.retrieve(query)
    assert res.has_sufficient_evidence is False
    assert res.insufficient_evidence_message is not None
