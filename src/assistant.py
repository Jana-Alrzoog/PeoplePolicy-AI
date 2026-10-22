"""Core Assistant coordinating Local Deterministic RAG, Validated Tools, and Gemini LLM."""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import (
    ORGANIZATION_NAME,
    POLICY_DISCLAIMER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    is_gemini_configured,
)
from src.retrieval import get_knowledge_base, RetrievalResult, RetrievedPassage
from src.safety import (
    check_out_of_scope,
    ToolCallCounter,
    SafeExecutionLog,
)
from src.tools import execute_tool

# Eastern to Western Arabic Numerals conversion table
EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


class AssistantResponse(BaseModel):
    """Structured response container for the assistant turn."""
    answer: str
    mode: str  # "local" or "gemini"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tool_result: Optional[Dict[str, Any]] = None
    audit_logs: List[SafeExecutionLog] = Field(default_factory=list)
    warning_message: Optional[str] = None
    is_out_of_scope: bool = False
    has_sufficient_evidence: bool = True


def normalize_digits(text: str) -> str:
    """Convert Eastern Arabic numerals to standard western digits."""
    return text.translate(EASTERN_ARABIC_DIGITS)


def extract_leave_calculation_params(query: str) -> Optional[Tuple[float, float]]:
    """Extract available balance and requested days from Arabic query text."""
    normalized = normalize_digits(query)

    # Specific patterns for balance and requested days
    # Examples: "عندي رصيد 15 يوم إجازة، وأبي آخذ 6 أيام"
    balance_match = re.search(
        r"(?:رصيد(?:ي)?|متوفر|متاح|عندي|لدي|املك|أملك)\s*(?:قدره|هو|منه|حوالي)?\s*(\d+(?:\.\d+)?)",
        normalized,
        re.IGNORECASE,
    )

    requested_match = re.search(
        r"(?:اخذ|آخذ|أخذ|اطلب|أطلب|طلب|اريد|أريد|ابي|أبي|استأذن)\s*(?:منه|منها)?\s*(\d+(?:\.\d+)?)",
        normalized,
        re.IGNORECASE,
    )

    if balance_match and requested_match:
        try:
            bal = float(balance_match.group(1))
            req = float(requested_match.group(1))
            return bal, req
        except ValueError:
            pass

    # Generic fallback: look for two numbers in context of leave
    if any(keyword in normalized for keyword in ["رصيد", "إجازة", "اجازة", "يتبقى", "باقي"]):
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", normalized)
        if len(numbers) >= 2:
            try:
                bal = float(numbers[0])
                req = float(numbers[1])
                return bal, req
            except ValueError:
                pass

    return None


class PeoplePolicyAssistant:
    """Main HR Policy Assistant engine supporting Local Deterministic & Gemini modes."""

    def __init__(self):
        self.kb = get_knowledge_base()
        self.tool_counter = ToolCallCounter()

    def _generate_local_answer(
        self,
        query: str,
        retrieval: RetrievalResult,
        tool_result: Optional[Dict[str, Any]],
    ) -> str:
        """Formulate a grounded, professional Arabic response deterministically."""
        parts: List[str] = []

        # If a tool calculation was performed, present the calculation clearly
        if tool_result and tool_result.get("success"):
            data = tool_result["data"]
            avail = data["available_balance"]
            req = data["requested_days"]
            rem = data["remaining_balance"]
            eligible = data["eligible"]
            reason = data["decision_reason"]

            if eligible:
                calc_summary = (
                    f"**نتيجة فحص الرصيد والحساب:**\n"
                    f"- **الرصيد المتاح:** {avail} يوماً\n"
                    f"- **الأيام المطلوبة:** {req} يوماً\n"
                    f"- **الرصيد المتبقي المتوقع:** {rem} يوماً\n"
                    f"- **حالة الأهلية المبدئية:** مؤهل حسابياً (نعم، يمكنك التقديم على الطلب)\n\n"
                    f"💡 *ملاحظة:* {reason}\n"
                    f"⚠️ **تنبيه إداري:** أهلية الرصيد الحسابية لا تعني الموافقة الإدارية النهائية؛ "
                    f"يجب رفع الطلب عبر بوابة الخدمة الذاتية واعتماده من مديرك المباشر وإدارة الموارد البشرية."
                )
            else:
                calc_summary = (
                    f"**نتيجة فحص الرصيد والحساب:**\n"
                    f"- **الرصيد المتاح:** {avail} يوماً\n"
                    f"- **الأيام المطلوبة:** {req} يوماً\n"
                    f"- **حالة الأهلية المبدئية:** غير مؤهل (لا يمكن قبول الطلب)\n\n"
                    f"❌ *سبب عدم الأهلية:* {reason}\n"
                    f"لا يُسمح بالحصول على رصيد سالب وفقاً لسياسة الإجازات السنوية في شركة أفق التجريبية."
                )
            parts.append(calc_summary)

        # Grounding in policy evidence
        if retrieval.has_sufficient_evidence and retrieval.passages:
            top_passage = retrieval.passages[0]
            # Extract main sentences for summary
            summary_snippet = top_passage.passage_text.splitlines()
            clean_lines = [l.strip() for l in summary_snippet if l.strip() and not l.startswith("#")]
            context_summary = " ".join(clean_lines[:4])

            policy_reference = (
                f"\n\n**المرجع من لائحة الموارد البشرية ({top_passage.source_id} — {top_passage.policy_title}):**\n"
                f"{context_summary}"
            )
            parts.append(policy_reference)

        elif not tool_result:
            parts.append(
                "لم يتم العثور على شواهد أو سياسات موثوقة ذات صلة بهذا الاستفسار في سياسات الشركة التجريبية."
            )

        return "\n".join(parts)

    def _call_gemini_api(
        self,
        query: str,
        retrieval: RetrievalResult,
        tool_result: Optional[Dict[str, Any]],
    ) -> str:
        """Generate response via Gemini SDK strictly grounded in retrieved evidence."""
        # Use google-genai official client
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build grounded context
        context_blocks = []
        for p in retrieval.passages:
            context_blocks.append(
                f"[المصدر: {p.source_id} - {p.policy_title}]\n{p.passage_text}"
            )
        grounded_context = "\n\n---\n\n".join(context_blocks)

        tool_context_str = ""
        if tool_result:
            tool_context_str = (
                f"\n\nنتيجة الأداة المحسوبة في بايثون بدقة Decimal (استخدم هذه الأرقام حصراً):\n"
                f"{tool_result}"
            )

        system_instruction = (
            f"أنت المساعد الذكي للموارد البشرية في {ORGANIZATION_NAME}.\n"
            f"تنبيه أساسي: {POLICY_DISCLAIMER}\n"
            "تعليمات صارمة:\n"
            "1. أجب باللغة العربية المهنية والفصحى فقط وبطريقة واضحة ومباشرة.\n"
            "2. استند حصراً إلى وثائق السياسات المرفقة فقط، ولا تخترع أي معلومات أو سياسات غير مذكورة.\n"
            "3. اذكر دائماً رمز المصدر الرسمي (مثل HR-LEAVE-001) الذي استندت إليه إجابتك.\n"
            "4. إذا تضمنت الإجابة نتيجة حساب رصيد، التزم تماماً بالأرقام الناتجة عن أداة بايثون ولا تغيرها.\n"
            "5. أكد دائماً أن أهلية الرصيد الحسابية ليست موافقة إدارية نهائية، وأن المساعد لا يملك صلاحية اعتماد الإجازات.\n"
            "6. إذا لم تتضمن الوثائق إجابة موثوقة، صرح بوضوح أنه لم يتم العثور على شواهد في سياسات الشركة."
        )

        prompt = (
            f"سياق الوثائق والسياسات المتاحة:\n{grounded_context}\n"
            f"{tool_context_str}\n\n"
            f"سؤال الموظف:\n{query}\n\n"
            "الإجابة المعتمدة باللغة العربية:"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )

        if not response.text:
            raise ValueError("Empty response received from Gemini model.")

        return response.text.strip()

    def process_query(self, query: str) -> AssistantResponse:
        """Process a user question through safety, retrieval, tool execution, and synthesis."""
        self.tool_counter.reset()
        audit_logs: List[SafeExecutionLog] = []

        # 1. Safety Boundary: Check out-of-scope queries
        is_oops, refusal_message = check_out_of_scope(query)
        if is_oops and refusal_message:
            return AssistantResponse(
                answer=refusal_message,
                mode="local",
                sources=[],
                tool_result=None,
                audit_logs=[],
                warning_message=None,
                is_out_of_scope=True,
                has_sufficient_evidence=False,
            )

        # 2. Local RAG Retrieval
        retrieval = self.kb.retrieve(query)

        # 3. Check for leave calculation intent
        tool_result: Optional[Dict[str, Any]] = None
        leave_params = extract_leave_calculation_params(query)

        if leave_params:
            avail_bal, req_days = leave_params
            if self.tool_counter.record_call():
                t_res, log_entry = execute_tool(
                    tool_name="calculate_leave_balance",
                    arguments={
                        "available_balance": avail_bal,
                        "requested_days": req_days,
                    },
                )
                tool_result = t_res
                audit_logs.append(log_entry)

        # Format sources list
        sources_list = [
            {
                "source_id": p.source_id,
                "policy_title": p.policy_title,
                "section_title": p.section_title,
                "similarity_score": p.similarity_score,
                "snippet": p.passage_text[:280] + ("..." if len(p.passage_text) > 280 else ""),
            }
            for p in retrieval.passages
        ]

        # 4. Synthesize Answer: Gemini (if configured) or Local Deterministic
        warning_msg: Optional[str] = None
        mode = "local"
        answer: str = ""

        # Check if Gemini is configured and we have supporting evidence or tool results
        if is_gemini_configured() and (retrieval.has_sufficient_evidence or tool_result):
            try:
                answer = self._call_gemini_api(
                    query=query,
                    retrieval=retrieval,
                    tool_result=tool_result,
                )
                mode = "gemini"
            except Exception as e:
                # Safe fallback to local deterministic mode
                warning_msg = (
                    f"تعذر الاتصال بنموذج Gemini ({type(e).__name__}). "
                    "تم الانتقال تلقائياً إلى الوضع المحلي الحتمي لضمان استمرارية الخدمة."
                )
                mode = "local"
                answer = self._generate_local_answer(
                    query=query,
                    retrieval=retrieval,
                    tool_result=tool_result,
                )
        else:
            # Deterministic Local Mode
            mode = "local"
            answer = self._generate_local_answer(
                query=query,
                retrieval=retrieval,
                tool_result=tool_result,
            )

        return AssistantResponse(
            answer=answer,
            mode=mode,
            sources=sources_list,
            tool_result=tool_result,
            audit_logs=audit_logs,
            warning_message=warning_msg,
            is_out_of_scope=False,
            has_sufficient_evidence=retrieval.has_sufficient_evidence,
        )


# Global assistant instance
_assistant_instance: Optional[PeoplePolicyAssistant] = None

def get_assistant() -> PeoplePolicyAssistant:
    """Return or initialize singleton Assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = PeoplePolicyAssistant()
    return _assistant_instance
