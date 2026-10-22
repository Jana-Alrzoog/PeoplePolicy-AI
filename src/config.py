"""Configuration and environment settings for PeoplePolicy AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load environment variables from .env file if available
load_dotenv(PROJECT_ROOT / ".env")

# Fictional Organization & Policy Constants
ORGANIZATION_NAME = "شركة أفق التجريبية"
POLICY_DISCLAIMER = "سياسة تجريبية مصطنعة لأغراض المشروع، وليست سياسة رسمية لأي جهة."

POLICY_CATALOG = {
    "HR-LEAVE-001": {
        "title": "سياسة الإجازات السنوية",
        "file": "annual_leave.md",
        "description": "استحقاق واحتساب رصيد الإجازات السنوية وشروط التقديم والترحيل.",
    },
    "HR-SICK-002": {
        "title": "سياسة الإجازات المرضية",
        "file": "sick_leave.md",
        "description": "استحقاق الإجازات المرضية والتقارير الطبية وسلّم الأجور.",
    },
    "HR-ATTEND-003": {
        "title": "سياسة الحضور وساعات العمل",
        "file": "attendance.md",
        "description": "ساعات العمل الرسمية، أيام العمل، فترة السماح، وفترات الراحة.",
    },
    "HR-REMOTE-004": {
        "title": "سياسة العمل عن بُعد",
        "file": "remote_work.md",
        "description": "أهلية العمل عن بعد، الحد الأقصى الأسبوعي، والضوابط التقنية.",
    },
    "HR-APPROVAL-005": {
        "title": "سياسة مسار اعتماد وتدفق الإجازات",
        "file": "leave_approval.md",
        "description": "دورة اعتماد الإجازات والحدود الصارمة للمساعد الذكي.",
    },
}

# RAG Engine settings
DEFAULT_TOP_K = 3
# Calibrated similarity threshold for Arabic TF-IDF cosine similarity
RELEVANCE_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.12"))

# Tool Execution & Security Limits
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "3"))
ALLOWED_TOOLS = frozenset(["calculate_leave_balance"])

# Gemini API settings (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def is_gemini_configured() -> bool:
    """Return True if a non-empty Gemini API key is configured in the environment."""
    return bool(GEMINI_API_KEY)
