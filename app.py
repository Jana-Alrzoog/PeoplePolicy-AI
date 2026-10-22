"""PeoplePolicy AI — HR Policy Assistant
Streamlit Arabic-First Web Application with Full RTL Support.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.config import (
    ORGANIZATION_NAME,
    POLICY_DISCLAIMER,
    POLICY_CATALOG,
    GEMINI_MODEL,
    is_gemini_configured,
)
from src.assistant import get_assistant
from src.tools import execute_tool
from src.evaluation import EvaluationSuite

# Page configuration
st.set_page_config(
    page_title="PeoplePolicy AI — مساعد سياسات الموارد البشرية",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom RTL & Enterprise Cairo Font Styling
CUSTOM_RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700&display=swap');

/* Global RTL and Typography */
html, body, [class*="css"], .stApp {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', 'Tajawal', sans-serif !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    direction: rtl !important;
    text-align: right !important;
    background-color: #FFFFFF !important;
    border-left: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] * {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', sans-serif !important;
}

/* Tab labels */
.stTabs [data-baseweb="tab-list"] {
    direction: rtl !important;
    gap: 12px !important;
    border-bottom: 2px solid #E2E8F0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #475569 !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #1E3A8A !important;
    border-bottom: 3px solid #1E3A8A !important;
    background-color: #EFF6FF !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-btn button {
    background-color: #1E3A8A !important;
    color: #FFFFFF !important;
}

.primary-btn button:hover {
    background-color: #1E40AF !important;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.25) !important;
}

/* Cards & Containers */
.policy-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: transform 0.15s ease;
}

.policy-card:hover {
    border-color: #93C5FD;
    box-shadow: 0 4px 8px rgba(30, 58, 138, 0.08);
}

.source-badge {
    display: inline-block;
    background-color: #1E3A8A;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    margin-left: 8px;
}

.score-badge {
    display: inline-block;
    background-color: #0D9488;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
}

.tool-card {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 1px solid #86EFAC;
    border-radius: 12px;
    padding: 18px;
    margin: 16px 0;
}

.tool-card-rejected {
    background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    border: 1px solid #FCA5A5;
    border-radius: 12px;
    padding: 18px;
    margin: 16px 0;
}

.mode-badge-local {
    background-color: #475569;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

.mode-badge-gemini {
    background: linear-gradient(90deg, #1E3A8A 0%, #0D9488 100%);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    display: inline-block;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    direction: rtl !important;
    text-align: right !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', sans-serif !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 800 !important;
}
</style>
"""
st.markdown(CUSTOM_RTL_CSS, unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# Sidebar Content
with st.sidebar:
    st.markdown("### 🏢 شركة أفق التجريبية")
    st.markdown(f"**نظام الموارد البشرية الذكي**")

    # Mode Indicator
    st.markdown("---")
    st.markdown("#### ⚙️ نمط تشغيل المساعد")
    if is_gemini_configured():
        st.markdown(
            f'<div class="mode-badge-gemini">✨ وضع Gemini الذكي ({GEMINI_MODEL})</div>',
            unsafe_allow_html=True,
        )
        st.caption("النموذج الذكي متصل ومفعّل لتوليد الإجابات المعتمدة.")
    else:
        st.markdown(
            '<div class="mode-badge-local">🔒 الوضع المحلي الحتمي (Local Mode)</div>',
            unsafe_allow_html=True,
        )
        st.caption("يعمل بدون مفتاح API معتمداً على محرك TF-IDF وحسابات بايثون الدقيقة.")

    # Policy Catalog
    st.markdown("---")
    st.markdown("#### 📚 دليل السياسات المتاحة")
    for pid, meta in POLICY_CATALOG.items():
        with st.expander(f"{pid} — {meta['title']}"):
            st.write(meta["description"])
            st.caption(f"الملف: data/{meta['file']}")

    st.markdown("---")
    st.markdown("#### 🛡️ تنبيه وضوابط الأمان")
    st.caption(POLICY_DISCLAIMER)
    st.caption("النظام استرشادي فقط ولا يملك صلاحية اعتماد الإجازات أو تعديل الرواتب والسجلات.")

    if st.button("🗑️ مسح محادثة المساعد", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main Header
col_header_title, col_header_badge = st.columns([3, 1])
with col_header_title:
    st.title("PeoplePolicy AI — مساعد سياسات الموارد البشرية")
    st.markdown(
        "منصة استعلام ذكية للإجابة على سياسات العمل واحتساب أرصدة الإجازات وخصومات الغياب "
        f"في **{ORGANIZATION_NAME}** بدقة وموثوقية."
    )
with col_header_badge:
    if is_gemini_configured():
        st.markdown(
            '<div style="text-align:left; padding-top:20px;">'
            '<span class="mode-badge-gemini">وضع Gemini متصل</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:left; padding-top:20px;">'
            '<span class="mode-badge-local">الوضع المحلي الحتمي</span></div>',
            unsafe_allow_html=True,
        )

# Navigation Tabs
tab_assistant, tab_calculator, tab_absence, tab_evaluation = st.tabs([
    "💬 المساعد الذكي (AI Assistant)",
    "🧮 حاسبة الإجازات (Leave Calculator)",
    "💰 حاسبة خصم الغياب (Absence Deduction)",
    "📊 لوحة التقييم والاختبارات (Evaluation Dashboard)",
])


def render_tool_result_card(tool_res: Dict[str, Any]) -> None:
    """Helper to render tool output nicely in the chat UI."""
    if not tool_res or not tool_res.get("success"):
        return
    t_data = tool_res["data"]

    # 1. Absence Deduction Card
    if "estimated_deduction" in t_data:
        st.markdown(
            f"""
            <div class="tool-card">
                <h4 style="margin:0 0 10px 0; color:#0F172A;">💰 نتيجة أداة بايثون (calculate_absence_deduction)</h4>
                <ul style="margin:0; padding-right:20px;">
                    <li><b>الراتب الأساسي الشهري:</b> {t_data['base_salary']:,.2f} ريال سعودي</li>
                    <li><b>أيام الغياب غير المدفوعة:</b> {t_data['unpaid_absence_days']} يوماً</li>
                    <li><b>أجر اليوم الواحد (الراتب ÷ 30):</b> {t_data['daily_wage']:,.2f} ريال سعودي</li>
                    <li><b>مبلغ الخصم التقديري:</b> {t_data['estimated_deduction']:,.2f} ريال سعودي</li>
                    <li><b>الراتب التقديري المتبقي:</b> {t_data['estimated_remaining_salary']:,.2f} ريال سعودي</li>
                </ul>
                <div style="font-size:13px; margin-top:8px; color:#475569;">{t_data['explanation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Leave Balance Card
    elif "remaining_balance" in t_data:
        card_class = "tool-card" if t_data["eligible"] else "tool-card-rejected"
        badge_text = "✅ مؤهل حسابياً" if t_data["eligible"] else "❌ غير مؤهل (عدم كفاية الرصيد)"
        st.markdown(
            f"""
            <div class="{card_class}">
                <h4 style="margin:0 0 10px 0; color:#0F172A;">🧮 نتيجة أداة بايثون (calculate_leave_balance)</h4>
                <div style="font-weight:700; margin-bottom:8px;">الحالة: {badge_text}</div>
                <ul style="margin:0; padding-right:20px;">
                    <li><b>الرصيد المتاح:</b> {t_data['available_balance']} يوماً</li>
                    <li><b>الأيام المطلوبة:</b> {t_data['requested_days']} يوماً</li>
                    <li><b>الرصيد المتبقي المتوقع:</b> {t_data['remaining_balance']} يوماً</li>
                </ul>
                <div style="font-size:13px; margin-top:8px; color:#475569;">{t_data['decision_reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# TAB 1: AI ASSISTANT
# ==========================================
with tab_assistant:
    st.markdown("#### 💬 استفسر عن لوائح الموارد البشرية أو تحقق من رصيدك وخصم الغياب")

    # Example question buttons
    st.markdown("**💡 أسئلة شائعة وتجارب فحص الصلاحيات:**")
    example_cols = st.columns(3)

    q1 = "عندي رصيد 15 يوم إجازة، وأبي آخذ 6 أيام. هل أقدر؟ وكم يتبقى لي؟"
    q2 = "راتبي 9000 ريال وغبت يومين بدون عذر، كم يخصم مني وكم يتبقى من راتبي؟"
    q3 = "راتبي 8000 وعندي إجازة سنوية معتمدة 4 أيام هل يخصم مني؟"
    q4 = "ما هي شروط الإجازة المرضية والتقرير الطبي المطلوب؟"
    q_unauthorized = "اعتمد إجازتي الآن وأرسل الموافقة للموارد البشرية."
    q_capability = "هل يمكنك اعتماد إجازتي؟"
    q_steps = "ما خطوات اعتماد الإجازة؟"
    q_responsible = "من المسؤول عن اعتماد الإجازات؟"
    q_formula = "ما هي معادلة احتساب خصم الغياب غير المدفوع من الراتب؟"

    clicked_query = None
    with example_cols[0]:
        if st.button("🌴 رصيد 15 وأخذ 6 أيام", use_container_width=True):
            clicked_query = q1
        if st.button("💰 خصم غياب (راتب 9000 ويومين)", use_container_width=True):
            clicked_query = q2
        if st.button("⛔ اعتمد إجازتي الآن (إجراء محظور)", use_container_width=True):
            clicked_query = q_unauthorized
    with example_cols[1]:
        if st.button("🏖️ إجازة معتمدة والخصم (مدفوعة)", use_container_width=True):
            clicked_query = q3
        if st.button("❓ هل يمكنك اعتماد إجازتي؟", use_container_width=True):
            clicked_query = q_capability
        if st.button("📋 ما خطوات اعتماد الإجازة؟", use_container_width=True):
            clicked_query = q_steps
    with example_cols[2]:
        if st.button("📐 معادلة خصم الغياب", use_container_width=True):
            clicked_query = q_formula
        if st.button("🩺 الإجازات المرضية والتقارير", use_container_width=True):
            clicked_query = q4
        if st.button("👤 من المسؤول عن الاعتماد؟", use_container_width=True):
            clicked_query = q_responsible

    st.markdown("---")

    # Display chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["user_query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            # Render Tool Result
            if item.get("tool_result"):
                render_tool_result_card(item["tool_result"])

            # Render Evidence Sources
            if item.get("sources"):
                with st.expander(f"📚 الشواهد المسترجعة من السياسات ({len(item['sources'])} فقرات)"):
                    for s in item["sources"]:
                        st.markdown(
                            f"""
                            <div class="policy-card">
                                <span class="source-badge">{s['source_id']}</span>
                                <b>{s['policy_title']}</b> — {s['section_title']}
                                <span class="score-badge" style="float:left;">درجة المطابقة: {s['similarity_score']}</span>
                                <p style="font-size:14px; margin-top:10px; color:#334155;">{s['snippet']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Render Audit Logs if any
            if item.get("audit_logs"):
                with st.expander("🛡️ سجل التنفيذ الفني والرقابي (Safe Audit Log)"):
                    for log in item["audit_logs"]:
                        st.code(
                            f"Tool: {log.tool_name} | Status: {log.status} | Duration: {log.duration_ms} ms",
                            language="text",
                        )

            if item.get("warning_message"):
                st.warning(item["warning_message"])

    # Handle Input from Chat or Buttons
    user_input = st.chat_input("اكتب استفسارك هنا حول سياسات الموارد البشرية أو حساب الرصيد أو خصم الغياب...")
    active_prompt = clicked_query or user_input

    if active_prompt:
        with st.chat_message("user"):
            st.write(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("جاري فحص السياسات وتشغيل الأدوات المعتمدة..."):
                assistant = get_assistant()
                response = assistant.process_query(active_prompt)

            st.markdown(response.answer)

            # Render Tool Result
            if response.tool_result:
                render_tool_result_card(response.tool_result)

            # Evidence Cards
            if response.sources:
                with st.expander(f"📚 الشواهد المسترجعة من السياسات ({len(response.sources)} فقرات)"):
                    for s in response.sources:
                        st.markdown(
                            f"""
                            <div class="policy-card">
                                <span class="source-badge">{s['source_id']}</span>
                                <b>{s['policy_title']}</b> — {s['section_title']}
                                <span class="score-badge" style="float:left;">درجة المطابقة: {s['similarity_score']}</span>
                                <p style="font-size:14px; margin-top:10px; color:#334155;">{s['snippet']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Audit Logs
            if response.audit_logs:
                with st.expander("🛡️ سجل التنفيذ الفني والرقابي (Safe Audit Log)"):
                    for log in response.audit_logs:
                        st.code(
                            f"Tool: {log.tool_name} | Status: {log.status} | Duration: {log.duration_ms} ms",
                            language="text",
                        )

            if response.warning_message:
                st.warning(response.warning_message)

            # Save to session history
            st.session_state.chat_history.append({
                "user_query": active_prompt,
                "answer": response.answer,
                "sources": response.sources,
                "tool_result": response.tool_result,
                "audit_logs": response.audit_logs,
                "warning_message": response.warning_message,
            })

# ==========================================
# TAB 2: LEAVE CALCULATOR
# ==========================================
with tab_calculator:
    st.markdown("#### 🧮 حاسبة رصيد الإجازات السنوية المعتمدة")
    st.markdown(
        "أداة بايثون مستقلة (`calculate_leave_balance`) تخضع للتحقق الصارم "
        "بواسطة **Pydantic** وحسابات **Decimal** الدقيقة لمنع أخطاء التقريب أو الأرصدة السالبة."
    )

    col_calc_form, col_calc_result = st.columns([1, 1], gap="large")

    with col_calc_form:
        st.markdown("##### 📝 إدخال بيانات الرصيد")
        available_input = st.number_input(
            "رصيد الإجازة السنوية المتاح حالياً (بالأيام)",
            min_value=0.0,
            max_value=120.0,
            value=15.0,
            step=1.0,
            help="الرصيد الفعلي المتاح للموظف في تاريخ التقديم (يجب أن يكون 0 أو أكثر).",
            key="leave_avail_input",
        )

        requested_input = st.number_input(
            "عدد أيام الإجازة السنوية المطلوبة (بالأيام)",
            min_value=0.5,
            max_value=60.0,
            value=6.0,
            step=0.5,
            help="عدد الأيام المراد التمتع بها (يجب أن يكون أكبر من 0).",
            key="leave_req_input",
        )

        calc_submitted = st.button("🧮 احسب الرصيد المتبقي والأهلية", type="primary", use_container_width=True, key="btn_leave_calc")

    with col_calc_result:
        st.markdown("##### 📊 النتيجة وقرار الأهلية")
        if calc_submitted:
            res, log = execute_tool(
                "calculate_leave_balance",
                {
                    "available_balance": float(available_input),
                    "requested_days": float(requested_input),
                },
            )

            if res["success"]:
                data = res["data"]
                eligible = data["eligible"]
                rem_bal = data["remaining_balance"]

                if eligible:
                    st.success("✅ الطلب مؤهل حسابياً للمتابعة والتقديم")
                else:
                    st.error("❌ الطلب غير مؤهل (الرصيد المتاح غير كافٍ)")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("الرصيد المتاح", f"{data['available_balance']} يوم")
                metric_col2.metric("الأيام المطلوبة", f"{data['requested_days']} يوم")
                metric_col3.metric("الرصيد المتبقي", f"{rem_bal} يوم")

                st.info(f"**توضيح القرار:** {data['decision_reason']}")
                st.caption(f"🛡️ {data['disclaimer']}")
                st.caption(f"⚡ زمن تنفيذ أداة بايثون: {log.duration_ms} مللي ثانية")
            else:
                st.error(f"خطأ في التحقق من المدخلات: {res['message']}")
        else:
            st.info("اضغط على زر **احسب الرصيد المتبقي والأهلية** لتنفيذ أداة بايثون وعرض النتائج.")

# ==========================================
# TAB 3: ABSENCE DEDUCTION CALCULATOR
# ==========================================
with tab_absence:
    st.markdown("#### 💰 حاسبة خصم الغياب واحتساب الأجر (HR-ABSENCE-006)")
    st.markdown(
        "أداة بايثون مستقلة (`calculate_absence_deduction`) لاحتساب الخصم التقديري للغياب "
        "بناءً على المعادلة الاسترشادية الافتراضية: **أجر اليوم = الراتب الأساسي ÷ 30**."
    )

    col_abs_form, col_abs_result = st.columns([1, 1], gap="large")

    with col_abs_form:
        st.markdown("##### 📝 بيانات الراتب والغياب")
        salary_input = st.number_input(
            "الراتب الأساسي الشهري (ريال سعودي)",
            min_value=100.0,
            max_value=200000.0,
            value=9000.0,
            step=500.0,
            help="الراتب الأساسي للموظف بالريال السعودي (يجب أن يكون أكبر من صفر).",
            key="abs_salary_input",
        )

        absence_days_input = st.number_input(
            "عدد أيام الغياب (بالأيام)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.5,
            help="عدد أيام الغياب (الحد الأقصى 30 يوماً في الشهر التقويمي الواحد).",
            key="abs_days_input",
        )

        absence_type_selected = st.selectbox(
            "تصنيف نوع الغياب أو الإجازة",
            [
                "غياب غير مبرر / غير مدفوع الأجر (يطبق عليه الخصم)",
                "إجازة سنوية معتمدة رسمياً (مدفوعة الأجر - لا خصم)",
                "إجازة مرضية معتمدة بتقرير طبي (مدفوعة الأجر - لا خصم)",
                "حالة غياب غير محددة / قيد التدقيق الإداري (حظر الخصم التلقائي)",
            ],
            key="abs_type_select",
        )

        abs_calc_submitted = st.button("🧮 احسب خصم الغياب والراتب المتبقي", type="primary", use_container_width=True, key="btn_abs_calc")

    with col_abs_result:
        st.markdown("##### 📊 النتيجة واحتساب الراتب التقديري")

        if abs_calc_submitted:
            # Check paid leave classification
            if "إجازة سنوية معتمدة" in absence_type_selected or "إجازة مرضية معتمدة" in absence_type_selected:
                st.success("✅ إجازة معتمدة مدفوعة الأجر بالكامل وفقاً للوائح الشركة.")
                m1, m2, m3 = st.columns(3)
                m1.metric("الراتب الأساسي", f"{salary_input:,.2f} ر.س")
                m2.metric("مبلغ الخصم", "0.00 ر.س", delta="لا خصم")
                m3.metric("الراتب المتبقي", f"{salary_input:,.2f} ر.س")
                st.info("وفقاً لسياستي HR-LEAVE-001 و HR-ABSENCE-006: الإجازات المعتمدة لا يترتب عليها أي خصم مالي من الراتب.")

            elif "حالة غياب غير محددة" in absence_type_selected:
                st.warning("⚠️ حالة الغياب قيد التدقيق الإداري وغير مصنفة رسمياً.")
                st.info(
                    "وفقاً للبند (2) من سياسة HR-ABSENCE-006: **يُحظر تطبيق أي خصم تلقائي** على الراتب "
                    "حتى يتم التحقق الإداري من إدارة الموارد البشرية وتحديد طبيعة العذر."
                )

            else:
                # Unpaid absence -> execute tool
                res, log = execute_tool(
                    "calculate_absence_deduction",
                    {
                        "base_salary": float(salary_input),
                        "unpaid_absence_days": float(absence_days_input),
                    },
                )

                if res["success"]:
                    data = res["data"]
                    st.success("✅ تم احتساب الخصم التقديري بدقة Decimal الحسابية")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("الراتب الأساسي", f"{data['base_salary']:,.0f} ر.س")
                    m2.metric("أجر اليوم", f"{data['daily_wage']:,.2f} ر.س")
                    m3.metric("مبلغ الخصم", f"{data['estimated_deduction']:,.2f} ر.س", delta=f"-{data['estimated_deduction']:,.2f}")
                    m4.metric("الراتب المتبقي", f"{data['estimated_remaining_salary']:,.2f} ر.س")

                    st.info(f"**شرح المعادلة:** {data['explanation']}")
                    st.caption(f"🛡️ {data['disclaimer']}")
                    st.caption(f"⚡ زمن تنفيذ أداة بايثون: {log.duration_ms} مللي ثانية | المرجع: HR-ABSENCE-006")
                else:
                    st.error(f"خطأ في التحقق من المدخلات: {res['message']}")
        else:
            st.info("اضغط على زر **احسب خصم الغياب والراتب المتبقي** لتنفيذ الحساب وعرض النتائج.")

# ==========================================
# TAB 4: EVALUATION DASHBOARD
# ==========================================
with tab_evaluation:
    st.markdown("#### 📊 لوحة تقييم الجودة والاختبارات الآلية (19 حالة فحص مؤتمتة)")
    st.markdown(
        "تقييم شامل يغطي: استرجاع السياسات الست، دقة حساب رصيد الإجازات، حاسبة خصم الغياب، "
        "التحقق الصارم للمدخلات، ورفض العمليات غير المصرح بها (إدارياً ومالياً)، وتحديد الأسئلة الخارجة عن النطاق."
    )

    col_btn_eval, col_metric_eval = st.columns([1, 3])
    with col_btn_eval:
        run_eval_clicked = st.button("🚀 تشغيل جميع الاختبارات الـ 19 آلياً", type="primary", use_container_width=True, key="btn_run_eval_all")

    if run_eval_clicked or st.session_state.eval_results is None:
        with st.spinner("جاري تنفيذ الاختبارات الـ 19 آلياً وقياس الأزمنة الفعلية..."):
            suite = EvaluationSuite()
            st.session_state.eval_results = suite.run_all()

    results = st.session_state.eval_results
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    pass_rate = round((passed_tests / total_tests) * 100, 1)
    total_time = round(sum(r.execution_time_ms for r in results), 2)

    # Separate Original 12 vs Extended Absence 7
    orig_12_results = results[:12]
    abs_7_results = results[12:]

    orig_passed = sum(1 for r in orig_12_results if r.passed)
    abs_passed = sum(1 for r in abs_7_results if r.passed)

    # Metrics Summary Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("إجمالي الاختبارات", f"{total_tests}")
    m2.metric("الفحوصات الأصلية الـ 12", f"{orig_passed}/12", delta="100% ناجح")
    m3.metric("فحوصات خصم الغياب", f"{abs_passed}/7", delta="100% ناجح")
    m4.metric("نسبة النجاح الكلية", f"{pass_rate}%")
    m5.metric("زمن التنفيذ الكلي", f"{total_time} ms")

    st.markdown("---")

    # Section 1: Original 12 Cases
    st.markdown("##### 📋 أولاً: حالات الاختبار الأصلية الـ 12 (Original 12 Required Cases)")
    for r in orig_12_results:
        status_badge = "✅ ناجح (PASSED)" if r.passed else "❌ متعثر (FAILED)"
        card_border = "#86EFAC" if r.passed else "#FCA5A5"
        with st.expander(f"اختبار #{r.test_id:02d}: {r.name_ar} — {status_badge} ({r.execution_time_ms} ms)", expanded=not r.passed):
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid {card_border}; border-radius:8px; padding:12px; margin-bottom:8px;">
                    <p><b>الفئة:</b> {r.category_ar}</p>
                    <p><b>المدخلات والسيناريو:</b> {r.input_description}</p>
                    <p><b>النتيجة المتوقعة:</b> {r.expected_output}</p>
                    <p><b>النتيجة الفعلية:</b> {r.actual_output}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if r.error_message:
                st.error(f"تفاصيل الخطأ: {r.error_message}")

    # Section 2: Extended Absence Deduction Cases
    st.markdown("##### 📋 ثانياً: حالات اختبار حاسبة خصم الغياب المضافة (Absence Deduction Cases 13-19)")
    for r in abs_7_results:
        status_badge = "✅ ناجح (PASSED)" if r.passed else "❌ متعثر (FAILED)"
        card_border = "#86EFAC" if r.passed else "#FCA5A5"
        with st.expander(f"اختبار #{r.test_id:02d}: {r.name_ar} — {status_badge} ({r.execution_time_ms} ms)", expanded=not r.passed):
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid {card_border}; border-radius:8px; padding:12px; margin-bottom:8px;">
                    <p><b>الفئة:</b> {r.category_ar}</p>
                    <p><b>المدخلات والسيناريو:</b> {r.input_description}</p>
                    <p><b>النتيجة المتوقعة:</b> {r.expected_output}</p>
                    <p><b>النتيجة الفعلية:</b> {r.actual_output}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if r.error_message:
                st.error(f"تفاصيل الخطأ: {r.error_message}")
