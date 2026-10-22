"""Security, governance, and agent boundary enforcement for PeoplePolicy AI."""

import time
import logging
import re
from typing import Any, Dict, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from src.config import ALLOWED_TOOLS, MAX_TOOL_CALLS

logger = logging.getLogger("peoplepolicy.safety")


class QueryIntent(str, Enum):
    """Categorized user intent for safety and authorization boundaries."""
    INFORMATIONAL = "INFORMATIONAL"         # Policy inquiry (e.g., "ما خطوات اعتماد الإجازة؟")
    ACTION_REQUEST = "ACTION_REQUEST"       # Command to execute unauthorized action (e.g., "اعتمد إجازتي...")
    CAPABILITY_QUERY = "CAPABILITY_QUERY"   # Inquiring if assistant can execute (e.g., "هل يمكنك اعتماد إجازتي؟")
    OUT_OF_SCOPE = "OUT_OF_SCOPE"           # Completely outside HR scope (e.g., "ما عاصمة فرنسا؟")


class SafeExecutionLog(BaseModel):
    """Sanitized technical audit log without sensitive data, API keys, or CoT."""
    tool_name: str
    status: str  # "SUCCESS", "REJECTED_UNAUTHORIZED", "VALIDATION_ERROR", "ERROR"
    duration_ms: float
    error_message: Optional[str] = None


class ToolCallCounter:
    """Tracks and limits the number of tool invocations within a single request turn."""

    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        self.max_calls = max_calls
        self.current_calls = 0

    def record_call(self) -> bool:
        if self.current_calls >= self.max_calls:
            return False
        self.current_calls += 1
        return True

    def reset(self) -> None:
        self.current_calls = 0


def is_tool_allowed(tool_name: str) -> bool:
    """Validate if requested tool name is in the strict security allowlist."""
    if not isinstance(tool_name, str):
        return False
    return tool_name.strip() in ALLOWED_TOOLS


def sanitize_log_message(message: str) -> str:
    """Remove potential secrets or accidental key leaks from logged messages."""
    redacted = str(message)
    # Mask common API key patterns if present
    redacted = re.sub(r"AIza[0-9A-Za-z\-_]+", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"bearer\s+[A-Za-z0-9\-_\.]+", "Bearer [REDACTED_TOKEN]", redacted, flags=re.IGNORECASE)
    return redacted


def create_safe_audit_entry(
    tool_name: str,
    status: str,
    start_time: float,
    error_message: Optional[str] = None,
) -> SafeExecutionLog:
    """Create a sanitized audit log entry showing execution duration and status only."""
    duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    sanitized_error = sanitize_log_message(error_message) if error_message else None
    return SafeExecutionLog(
        tool_name=tool_name,
        status=status,
        duration_ms=duration_ms,
        error_message=sanitized_error,
    )


# Common out-of-scope triggers (e.g. general geography, pop culture, non-HR programming)
OUT_OF_SCOPE_PATTERNS = [
    r"عاصمة\s+(فرنسا|مصر|أمريكا|بريطانيا|اليابان|ألمانيا|إيطاليا)",
    r"من\s+(فاز|ربح)\s+(بكأس|بالدوري|بالمباراة)",
    r"اكتب\s+لي\s+كود\s+(جافا|بايثون|سي)",
    r"ما\s+هو\s+الطقس",
    r"من\s+هو\s+رئيس",
]

def check_out_of_scope(query: str) -> Tuple[bool, Optional[str]]:
    """Detect if the user query is clearly outside the HR and workplace policies scope."""
    cleaned = query.strip()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            arabic_refusal = (
                "عذراً، أنا مساعد ذكي مخصص حصرياً لسياسات الموارد البشرية وإجراءات الإجازات "
                "في شركة أفق التجريبية. لا يمكنني الإجابة على استفسارات عامة خارج هذا النطاق."
            )
            return True, arabic_refusal
    return False, None


# ==============================================================================
# AUTHORIZATION & INTENT CLASSIFICATION (Distinguish Info vs Action vs Capability)
# ==============================================================================

# Informational framing regexes (Question starters seeking knowledge/policy details)
INFORMATIONAL_PREFIXES = [
    r"^ما\s+(خطوات|طريقه|اليه|شروط|سياسه|اجراءات|مراحل|ضوابط|معادله)",
    r"^ما\s+هي\s+(خطوات|طريقه|اليه|شروط|سياسه|اجراءات|مراحل|ضوابط|معادله)",
    r"^ما\s+هو\s+(مسار|نظام|اجراء|دور|ترتيب|خصم)",
    r"^من\s+(المسؤول|الذي|المخول|صاحب|الجهه)",
    r"^كيف\s+(يتم|تتم|اقدم|استطيع|يحصل|يجري|احسب|يحتسب)",
    r"^هل\s+(يشترط|يلزم|يحق|يتطلب|يجوز|يخصم|يتم\s+خصم|الغياب\s+يخصم|الاجازه|الاجازات)",
    r"^كم\s+(يخصم|الخصم|يتبقي|اقتطاع|مبلغ)",
    r"^احسب\s+(لي\s+)?(خصم|الخصم|راتب|الراتب|رصيد|الرصيد)",
    r"^شرح\s+(مسار|خطوات|اليه|دوره|طريقه|معادله)",
    r"^وضح\s+(لي\s+)?(مسار|خطوات|اليه|سياسه|معادله)",
]

# Capability inquiry patterns (User asks whether the assistant CAN execute an action)
CAPABILITY_PATTERNS = [
    r"^(هل\s+)?(يمكنك|تقدر|تستطيع|بامكانك|عندك\s+صلاحيه|لديك\s+صلاحيه|في\s+امكانك)\s+(اعتماد|الموافقه|رفض|ارسال|تعديل|حذف|خصم|توقيع)",
    r"^(هل\s+)?(انت\s+)?(تعتمد|توافق|ترسل|تحذف|تعدل|تخصم)\s+(اجازتي|طلبي|الاجازات|الراتب|راتبي)",
    r"^(هل\s+)?(تقوم|تستطيع\s+ان\s+تقوم)\s+(باعتماد|بالموافقه|بارسال|بتعديل|بخصم)",
    r"^(هل\s+)?(المساعد|النظام)\s+(يعتمد|يوافق|يرسل|يعدل|يخصم)",
]

# Action execution command patterns (Imperatives or explicit execution requests)
ACTION_COMMAND_PATTERNS = [
    (r"\b(اعتمد|اعتمدي|وافق\s+علي|وافق)\s+(اجازتي|طلبي|الاجازه|الطلب)\b", "approve_leave"),
    (r"\b(ارسل|ابعت)\s+(الموافقه|الاعتماد|الايميل|البريد|الاشعار)\b", "send_email"),
    (r"\b(احذف|امسح)\s+(سجل|بيانات|موظف|ملف)\b", "delete_employee"),
    (r"\b(عدل|غير|حدث)\s+(رصيدي|الرصيد|الراتب|البيانات)\b", "modify_records"),
    (r"\b(اخصم|خصم|وقع\s+خصم|نزل\s+خصم|سجل\s+خصم)\s+(من\s+راتب(?:ي)?|علي\s+راتب(?:ي)?|من\s+المسير|الراتب|مبلغ)\b", "modify_payroll"),
    (r"\b(عاقب|جازي|طبق\s+جزاء|فرض\s+عقوبه|وقع\s+جزاء)\b", "impose_penalty"),
]


def _normalize_arabic_for_intent(text: str) -> str:
    """Light normalization for intent matching."""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة\b", "ه", text)
    text = re.sub(r"ى\b", "ي", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_action_intent(query: str) -> Tuple[QueryIntent, Optional[str], Optional[str]]:
    """
    Distinguish informational questions from actual execution requests and capability queries.
    
    Returns:
        (QueryIntent, target_action, user_message)
    """
    normalized = _normalize_arabic_for_intent(query)

    # 1. Check if it's an informational question first (takes precedence over keyword hits)
    for pattern in INFORMATIONAL_PREFIXES:
        if re.search(pattern, normalized):
            return QueryIntent.INFORMATIONAL, None, None

    # 2. Check if it's a capability/permission question about the assistant
    for pattern in CAPABILITY_PATTERNS:
        if re.search(pattern, normalized):
            explanation = (
                "**توضيح حدود وصلاحيات المساعد:**\n"
                "**لا، لا يمكنني اعتماد إجازتك أو اتخاذ أي إجراء إداري نيابة عنك.**\n"
                "أنا مساعد ذكي معلوماتي واستشاري فقط، ولا أملك صلاحية اعتماد طلبات الإجازة، "
                "أو إرسال رسائل أو موافقات إدارية، أو تعديل السجلات والبيانات في شركة أفق التجريبية.\n\n"
                "وفقاً للائحة المعتمدة، يجب على الموظف رفع الطلب عبر بوابة الخدمة الذاتية الرسمية، "
                "ليتولى المدير المباشر مراجعته واعتماده، ثم توثيقه لدى إدارة عمليات الموارد البشرية."
            )
            return QueryIntent.CAPABILITY_QUERY, "approve_leave", explanation

    # 3. Check if it's an unauthorized action execution command
    for pattern, action_name in ACTION_COMMAND_PATTERNS:
        if re.search(pattern, normalized):
            if action_name in ["modify_payroll", "impose_penalty"]:
                refusal = (
                    "⛔ **رفض إجرائي — عملية غير مصرح بها:**\n"
                    "عذراً، بصفتي مساعداً ذكياً للموارد البشرية، **لا يمكنني تعديل مسيرات الرواتب، "
                    "أو فرض جزاءات مالية، أو توقيع خصومات رسمية، أو تعديل السجلات المالية للموظفين** في شركة أفق التجريبية.\n\n"
                    "هذا النظام استرشادي معلوماتي فقط، وتختص إدارة الرواتب وعمليات الموارد البشرية حصرياً بتسوية الرواتب وإقرار الخصومات الرسمية."
                )
            else:
                refusal = (
                    "⛔ **رفض إجرائي — عملية غير مصرح بها:**\n"
                    "عذراً، بصفتي مساعداً ذكياً للموارد البشرية، **لا يمكنني اعتماد طلبات الإجازة، "
                    "أو إرسال رسائل أو موافقات إدارية، أو تعديل السجلات والبيانات الوظيفية** في شركة أفق التجريبية.\n\n"
                    "هذا النظام مخصص لتقديم المعلومات الاسترشادية واحتساب الأرصدة فقط، "
                    "ولا يملك أي صلاحيات تنفيذية في دورة العمل الإدارية.\n\n"
                    "يرجى تقديم طلب الإجازة عبر بوابة الخدمة الذاتية الرسمية للموارد البشرية "
                    "ليتم اعتماده وفق مسار الاعتماد الإداري النظامي من مديرك المباشر وإدارة الموارد البشرية."
                )
            return QueryIntent.ACTION_REQUEST, action_name, refusal

    # Default to informational query
    return QueryIntent.INFORMATIONAL, None, None


def sanitize_prompt_injection(content: str) -> str:
    """Defensive sanitation to ensure retrieved text is treated strictly as passive context."""
    markers = [
        "SYSTEM:", "INSTRUCTION:", "IGNORE PREVIOUS", "تجاهل التعليمات", 
        "أنت الآن", "YOU ARE NOW", "EXECUTE:", "ADMIN_OVERRIDE"
    ]
    sanitized = content
    for marker in markers:
        sanitized = sanitized.replace(marker, "[FILTERED_OVERRIDE]")
    return sanitized
