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
    classify_action_intent,
    create_safe_audit_entry,
    is_tool_allowed,
    QueryIntent,
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
    intent: str = "INFORMATIONAL"
    is_action_refusal: bool = False


def normalize_digits(text: str) -> str:
    """Convert Eastern Arabic numerals to standard western digits."""
    return text.translate(EASTERN_ARABIC_DIGITS)


def extract_leave_calculation_params(query: str) -> Optional[Tuple[float, float]]:
    """Extract available balance and requested days from Arabic query text."""
    normalized = normalize_digits(query)

    # If the query is specifically about salary/absence deduction, do not confuse with annual leave
    if any(k in normalized for k in ["راتب", "أجر", "خصم غياب", "يخصم من"]):
        return None

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


def extract_absence_deduction_params(query: str) -> Optional[Tuple[float, float, str]]:
    """
    Extract base salary, absence days, and classification status.
    Status can be: 'unpaid', 'approved_paid', or 'unknown'.
    Returns (salary, absence_days, status) or None.
    """
    cleaned = normalize_digits(query).replace("،", " ").replace(",", "")

    # Check if query is related to salary and absence deduction
    absence_keywords = ["راتب", "راتبي", "أجر", "أجري", "يخصم", "خصم", "غياب", "غبت", "اقتطاع"]
    if not any(k in cleaned for k in absence_keywords):
        return None

    # Determine classification status
    status = "unpaid"
    if any(w in cleaned for w in ["غير مدفوع", "غير مدفوعة", "بدون عذر", "غير مبرر", "غياب غير مدفوع"]):
        status = "unpaid"
    elif any(w in cleaned for w in ["لا أعلم", "لا اعلم", "لا أدري", "لا ادري", "مجهول", "غير مصنف", "قيد التدقيق", "بدون تحديد", "لم يحدد"]):
        status = "unknown"
    elif any(w in cleaned for w in ["إجازة سنوية", "اجازة سنوية", "إجازة اعتيادية", "اجازة معتمدة", "إجازة مرضية", "مرضية موثقة", "إجازة مدفوعة", "معتمدة"]):
        status = "approved_paid"

    # Extract Salary
    salary = None
    sal_match = re.search(
        r"(?:راتب(?:ي)?|أجر(?:ي)?|الراتب)\s*(?:الأساسي)?\s*(?:هو|قدره|حوالي)?\s*(\d+(?:\.\d+)?)",
        cleaned,
        re.IGNORECASE,
    )
    if not sal_match:
        sal_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ريال|ر\.س|رس|SAR)\s*(?:راتب|راتبي)?", cleaned)

    if sal_match:
        try:
            salary = float(sal_match.group(1))
        except ValueError:
            pass

    # Extract Absence Days
    days = None
    if "يومين" in cleaned or "يومان" in cleaned:
        days = 2.0
    elif "يوم واحد" in cleaned or "يوما واحدا" in cleaned:
        days = 1.0
    else:
        days_match = re.search(r"(?:غبت|غياب|انقطعت|انقطاع)\s*(?:عن العمل|عنه)?\s*(\d+(?:\.\d+)?)", cleaned)
        if not days_match:
            days_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:يوم|ايام|أيام)\s*(?:غياب|بدون عذر|غير مدفوع)?", cleaned)
        if days_match:
            try:
                days = float(days_match.group(1))
            except ValueError:
                pass

    if salary is not None and days is not None:
        return salary, days, status

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
        intent: QueryIntent = QueryIntent.INFORMATIONAL,
        refusal_prefix: Optional[str] = None,
        status_override_message: Optional[str] = None,
    ) -> str:
        """Formulate a grounded, professional Arabic response deterministically."""
        parts: List[str] = []

        # 1. If this is an unauthorized action request or capability inquiry, lead with the explicit refusal/clarification
        if refusal_prefix:
            parts.append(refusal_prefix)

        # 2. If an override message exists (e.g. approved paid leave or unknown status)
        if status_override_message:
            parts.append(status_override_message)

        # 3. If a calculation tool was executed, present calculation results clearly
        if tool_result and tool_result.get("success"):
            data = tool_result["data"]

            # Absence Deduction Tool Result
            if "estimated_deduction" in data:
                salary = data["base_salary"]
                days = data["unpaid_absence_days"]
                daily = data["daily_wage"]
                deduction = data["estimated_deduction"]
                remaining = data["estimated_remaining_salary"]
                explanation = data["explanation"]
                disclaimer = data["disclaimer"]

                calc_summary = (
                    f"**نتيجة احتساب الخصم التقديري للغياب (HR-ABSENCE-006):**\n"
                    f"- **الراتب الأساسي الشهري:** {salary:,.2f} ريال سعودي\n"
                    f"- **أيام الغياب غير المدفوعة:** {days} يوماً\n"
                    f"- **أجر اليوم الواحد (الراتب ÷ 30):** {daily:,.2f} ريال سعودي\n"
                    f"- **مبلغ الخصم التقديري:** {deduction:,.2f} ريال سعودي\n"
                    f"- **الراتب التقديري المتبقي:** {remaining:,.2f} ريال سعودي\n\n"
                    f"💡 *شرح المعادلة:* {explanation}\n\n"
                    f"⚠️ **إخلاء مسؤولية وضوابط:** {disclaimer} "
                    f"المساعد الذكي أداة استرشادية ولا يملك صلاحية تعديل مسيرات الرواتب أو فرض جزاءات مالية."
                )
                parts.append(calc_summary)

            # Leave Balance Calculation Result
            elif "remaining_balance" in data:
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

        # 4. Grounding in policy evidence
        if retrieval.has_sufficient_evidence and retrieval.passages:
            top_passage = retrieval.passages[0]
            norm_q = query.replace("؟", "").strip()

            # Dedicated direct answers for HR-ABSENCE-006 questions
            if top_passage.source_id == "HR-ABSENCE-006" and intent == QueryIntent.INFORMATIONAL:
                if any(w in norm_q for w in ["معادلة", "معادله", "كيف يحسب", "طريقة حساب", "حساب الخصم"]):
                    informational_answer = (
                        "**معادلة احتساب خصم الغياب غير المدفوع (HR-ABSENCE-006):**\n"
                        "1. **أجر اليوم الواحد:** الراتب الأساسي الشهري ÷ 30 يوماً.\n"
                        "2. **الخصم التقديري:** أجر اليوم الواحد × عدد أيام الغياب غير المدفوعة.\n"
                        "3. **الراتب التقديري المتبقي:** الراتب الأساسي الشهري − إجمالي الخصم التقديري.\n"
                        "⚠️ *تنبيه:* الإجازات السنوية والمرضية المعتمدة مدفوعة الأجر بالكامل ولا يُخصم منها، "
                        "ولا يجوز أن يتجاوز الخصم إجمالي الراتب الأساسي."
                    )
                    parts.append(informational_answer)

            # Dedicated direct answers for HR-APPROVAL-005 questions
            elif top_passage.source_id == "HR-APPROVAL-005" and intent == QueryIntent.INFORMATIONAL:
                if any(w in norm_q for w in ["خطوات", "مسار", "كيف"]):
                    informational_answer = (
                        "**خطوات مسار اعتماد الإجازة في شركة أفق التجريبية:**\n"
                        "1. **الخطوة الأولى (التقديم):** يقوم الموظف برفع طلب الإجازة إلكترونياً عبر بوابة الخدمة الذاتية للموارد البشرية مع تحديد التواريخ والموظف البديل.\n"
                        "2. **الخطوة الثانية (مراجعة واعتماد المدير المباشر):** يتلقى المدير المباشر إشعاراً آلياً لمراجعة الطلب واتخاذ قرار القبول أو الرفض خلال مدة أقصاها 48 ساعة عمل.\n"
                        "3. **الخطوة الثالثة (توثيق الموارد البشرية):** بعد موافقة المدير، تتولى إدارة عمليات الموارد البشرية مطابقة الرصيد وتوثيق الإجازة في السجل الوظيفي وإصدار الإشعار النهائي للموظف."
                    )
                    parts.append(informational_answer)
                elif any(w in norm_q for w in ["المسؤول", "من يعتمد", "صلاحية"]):
                    informational_answer = (
                        "**الجهة المسؤولة عن اعتماد الإجازات:**\n"
                        "- **المدير المباشر:** هو المسؤول عن مراجعة الطلب واتخاذ قرار الاعتماد الإداري أو الرفض خلال 48 ساعة عمل.\n"
                        "- **إدارة عمليات الموارد البشرية:** هي المسؤولة عن مطابقة الرصيد وتوثيق الإجازة في السجل الوظيفي للموظف.\n"
                        "⚠️ *تنبيه:* المساعد الذكي لا يملك أي صلاحيات لاعتماد الإجازات أو تعديل السجلات."
                    )
                    parts.append(informational_answer)

            # Context summary citation
            summary_snippet = top_passage.passage_text.splitlines()
            clean_lines = [l.strip() for l in summary_snippet if l.strip() and not l.startswith("#")]
            context_summary = " ".join(clean_lines[:4])

            policy_reference = (
                f"\n\n**المرجع المعتمد من لائحة الموارد البشرية ({top_passage.source_id} — {top_passage.policy_title}):**\n"
                f"{context_summary}"
            )
            parts.append(policy_reference)

        elif not tool_result and not refusal_prefix and not status_override_message:
            parts.append(
                "لم يتم العثور على شواهد أو سياسات موثوقة ذات صلة بهذا الاستفسار في سياسات الشركة التجريبية."
            )

        return "\n\n".join(parts)

    def _call_gemini_api(
        self,
        query: str,
        retrieval: RetrievalResult,
        tool_result: Optional[Dict[str, Any]],
        intent: QueryIntent = QueryIntent.INFORMATIONAL,
        refusal_prefix: Optional[str] = None,
    ) -> str:
        """Generate response via Gemini SDK strictly grounded in retrieved evidence."""
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

        safety_instruction = ""
        if intent == QueryIntent.ACTION_REQUEST:
            safety_instruction = (
                "⚠️ تنبيه أمني حرج: المستخدم يطلب تنفيذ إجراء إداري أو مالي (مثل اعتماد إجازة أو تعديل راتب)! "
                "يجب عليك أن ترفض الطلب رفضاً صريحاً وقاطعاً، وأن توضح أنك مساعد استرشادي لا تملك صلاحية اعتماد الإجازات "
                "أو تعديل مسيرات الرواتب أو فرض جزاءات مالية، وأن تشير إلى السياسة المعتمدة.\n"
            )
        elif intent == QueryIntent.CAPABILITY_QUERY:
            safety_instruction = (
                "⚠️ تنبيه أمني: المستخدم يسأل عن قدرتك على اعتماد الإجازة أو تعديل الرواتب. "
                "أكد صراحة وبوضوح تام أنك لا تستطيع تنفيذ ذلك وأن ذلك من اختصاص القنوات الرسمية.\n"
            )

        system_instruction = (
            f"أنت المساعد الذكي للموارد البشرية في {ORGANIZATION_NAME}.\n"
            f"تنبيه أساسي: {POLICY_DISCLAIMER}\n"
            f"{safety_instruction}"
            "تعليمات صارمة:\n"
            "1. أجب باللغة العربية المهنية والفصحى فقط وبطريقة واضحة ومباشرة.\n"
            "2. استند حصراً إلى وثائق السياسات المرفقة فقط، ولا تخترع أي معلومات أو سياسات غير مذكورة.\n"
            "3. اذكر دائماً رمز المصدر الرسمي (مثل HR-ABSENCE-006 أو HR-LEAVE-001) الذي استندت إليه إجابتك.\n"
            "4. إذا تضمنت الإجابة نتيجة حساب أداة بايثون، التزم تماماً بالأرقام الناتجة ولا تغيرها.\n"
            "5. المساعد الذكي لا يملك أي صلاحيات تنفيذية، ولا يعتمد إجازات، ولا يعدل رواتب، ولا يوقع جزاءات ملزمة.\n"
            "6. فرق دائماً بين شرح السياسة وبين رفض تنفيذ الإجراء عند طلبه.\n"
            "7. ميز بين الإجازات المعتمدة (مدفوعة الأجر) وبين الغياب غير المدفوع.\n"
            "8. إذا لم تتضمن الوثائق إجابة موثوقة، صرح بوضوح أنه لم يتم العثور على شواهد في سياسات الشركة."
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

        generated = response.text.strip()

        # Failsafe verification
        if intent == QueryIntent.ACTION_REQUEST and not any(
            phrase in generated for phrase in ["لا يمكنني", "لا أملك صلاحية", "لا استطيع", "غير مصرح", "رفض"]
        ):
            generated = f"{refusal_prefix}\n\n{generated}"

        return generated

    def process_query(self, query: str) -> AssistantResponse:
        """Process a user question through safety, retrieval, tool execution, and synthesis."""
        start_time = time.perf_counter()
        self.tool_counter.reset()
        audit_logs: List[SafeExecutionLog] = []

        # 1. Safety Boundary: Check out-of-scope queries
        is_oops, oops_refusal = check_out_of_scope(query)
        if is_oops and oops_refusal:
            return AssistantResponse(
                answer=oops_refusal,
                mode="local",
                sources=[],
                tool_result=None,
                audit_logs=[],
                warning_message=None,
                is_out_of_scope=True,
                has_sufficient_evidence=False,
                intent=QueryIntent.OUT_OF_SCOPE.value,
                is_action_refusal=False,
            )

        # 2. Safety & Authorization: Classify Intent
        intent, target_action, refusal_message = classify_action_intent(query)
        is_action_refusal = (intent == QueryIntent.ACTION_REQUEST)

        # Enforce Tool Authorization: If an unauthorized action command was attempted
        if intent == QueryIntent.ACTION_REQUEST:
            action_name = target_action or "unauthorized_action"
            if not is_tool_allowed(action_name):
                audit_log = create_safe_audit_entry(
                    tool_name=action_name,
                    status="REJECTED_UNAUTHORIZED",
                    start_time=start_time,
                    error_message=f"محاولة غير مصرح بها لتنفيذ إجراء إداري محظور: {action_name}",
                )
                audit_logs.append(audit_log)

        # 3. Local RAG Retrieval
        retrieval = self.kb.retrieve(query)

        # 4. Check for Calculation Intents (Absence Deduction OR Leave Balance)
        tool_result: Optional[Dict[str, Any]] = None
        status_override_message: Optional[str] = None

        # Check Absence Deduction Intent
        absence_params = extract_absence_deduction_params(query)
        if absence_params and intent != QueryIntent.ACTION_REQUEST:
            salary, absence_days, leave_status = absence_params

            if leave_status == "approved_paid":
                # Approved paid leave -> NO deduction
                status_override_message = (
                    "**توضيح استحقاق الإجازة المعتمدة (لا ينطبق أي خصم):**\n"
                    "وفقاً لسياسة الإجازات وسياسة احتساب الأجر (HR-LEAVE-001 و HR-ABSENCE-006): "
                    "تعتبر الإجازات السنوية المعتمدة رسمياً والإجازات المرضية الموثقة بتقرير طبي معتمد "
                    "**إجازات مدفوعة الأجر بالكامل** ولا يترتب عليها أي خصم من الراتب.\n"
                    f"- **الراتب الأساسي:** {salary:,.2f} ريال سعودي\n"
                    f"- **أيام الإجازة المعتمدة:** {absence_days} يوماً\n"
                    f"- **مبلغ الخصم المستحق:** 0.00 ريال سعودي\n"
                    f"- **الراتب المتبقي:** {salary:,.2f} ريال سعودي بدون أي اقتطاع."
                )
            elif leave_status == "unknown":
                # Classification is unknown -> DO NOT automatically apply deduction!
                status_override_message = (
                    "**تنبيه نظامي بخصوص حالة الغياب غير المحددة (عدم تطبيق خصم تلقائي):**\n"
                    "وفقاً للبند (2) من سياسة خصم الغياب واحتساب الأجر (HR-ABSENCE-006): "
                    "إذا كانت حالة الغياب **غير مصنفة، أو مجهولة السبب، أو قيد التدقيق الإداري**، "
                    "**لا يجوز تطبيق أي خصم تلقائي على الراتب**.\n"
                    "يجب التحقق الرسمي أولاً من قبل إدارة الموارد البشرية والمدير المباشر لتحديد ما إذا كان الغياب "
                    "مبرراً بعذر مشروع أو غياباً غير مدفوع الأجر قبل إقرار أي خصم في مسير الرواتب."
                )
            else:
                # Unpaid absence -> execute calculate_absence_deduction
                if self.tool_counter.record_call():
                    t_res, log_entry = execute_tool(
                        tool_name="calculate_absence_deduction",
                        arguments={
                            "base_salary": salary,
                            "unpaid_absence_days": absence_days,
                        },
                    )
                    tool_result = t_res
                    audit_logs.append(log_entry)

        # Check Leave Balance Intent (if not absence deduction)
        elif intent != QueryIntent.ACTION_REQUEST:
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

        # 5. Synthesize Answer: Gemini (if configured) or Local Deterministic
        warning_msg: Optional[str] = None
        mode = "local"
        answer: str = ""

        if is_gemini_configured() and (retrieval.has_sufficient_evidence or tool_result or refusal_message):
            try:
                answer = self._call_gemini_api(
                    query=query,
                    retrieval=retrieval,
                    tool_result=tool_result,
                    intent=intent,
                    refusal_prefix=refusal_message,
                )
                mode = "gemini"
            except Exception as e:
                warning_msg = (
                    f"تعذر الاتصال بنموذج Gemini ({type(e).__name__}). "
                    "تم الانتقال تلقائياً إلى الوضع المحلي الحتمي لضمان استمرارية الخدمة."
                )
                mode = "local"
                answer = self._generate_local_answer(
                    query=query,
                    retrieval=retrieval,
                    tool_result=tool_result,
                    intent=intent,
                    refusal_prefix=refusal_message,
                    status_override_message=status_override_message,
                )
        else:
            mode = "local"
            answer = self._generate_local_answer(
                query=query,
                retrieval=retrieval,
                tool_result=tool_result,
                intent=intent,
                refusal_prefix=refusal_message,
                status_override_message=status_override_message,
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
            intent=intent.value,
            is_action_refusal=is_action_refusal,
        )


# Global assistant instance
_assistant_instance: Optional[PeoplePolicyAssistant] = None

def get_assistant() -> PeoplePolicyAssistant:
    """Return or initialize singleton Assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = PeoplePolicyAssistant()
    return _assistant_instance
