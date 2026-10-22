"""PeoplePolicy AI — HR Policy Assistant
Streamlit Arabic-First Web Application with Full RTL Support.
"""

import sys
from pathlib import Path

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
.stTextInput input, .stNumberInput input {
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
    st.caption("النظام استرشادي فقط ولا يملك صلاحية اعتماد الإجازات أو تعديل السجلات.")

    if st.button("🗑️ مسح محادثة المساعد", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main Header
col_header_title, col_header_badge = st.columns([3, 1])
with col_header_title:
    st.title("PeoplePolicy AI — مساعد سياسات الموارد البشرية")
    st.markdown(
        "منصة استعلام ذكية للإجابة على سياسات العمل واحتساب أرصدة الإجازات السنوية "
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
tab_assistant, tab_calculator, tab_evaluation = st.tabs([
    "💬 المساعد الذكي (AI Assistant)",
    "🧮 حاسبة الإجازات (Leave Calculator)",
    "📊 لوحة التقييم والاختبارات (Evaluation Dashboard)",
])

# ==========================================
# TAB 1: AI ASSISTANT
# ==========================================
with tab_assistant:
    st.markdown("#### 💬 استفسر عن لوائح الموارد البشرية أو تحقق من رصيدك")

    # Example question buttons
    st.markdown("**💡 أسئلة شائعة جاهزة للتجربة السريعة:**")
    example_cols = st.columns(3)
    
    q1 = "عندي رصيد 15 يوم إجازة، وأبي آخذ 6 أيام. هل أقدر؟ وكم يتبقى لي؟"
    q2 = "ما هي ساعات العمل الرسمية وفترة السماح للحضور الصباحي؟"
    q3 = "كم يوماً يسمح بالعمل عن بعد أسبوعياً وما هي شروطه؟"
    q4 = "ما هي شروط الإجازة المرضية والتقرير الطبي المطلوب؟"
    q5 = "ما هو مسار اعتماد الإجازة وهل يستطيع المساعد اعتمادها؟"
    q6 = "عندي رصيد 4 أيام إجازة وأريد طلب 10 أيام، هل أستطيع؟"

    clicked_query = None
    with example_cols[0]:
        if st.button("🌴 رصيد 15 وأخذ 6 أيام", use_container_width=True):
            clicked_query = q1
        if st.button("🩺 الإجازات المرضية والتقارير", use_container_width=True):
            clicked_query = q4
    with example_cols[1]:
        if st.button("⏰ مواعيد الدوام وفترة السماح", use_container_width=True):
            clicked_query = q2
        if st.button("📋 مسار اعتماد الإجازة", use_container_width=True):
            clicked_query = q5
    with example_cols[2]:
        if st.button("💻 سياسة العمل عن بُعد", use_container_width=True):
            clicked_query = q3
        if st.button("⚠️ رصيد 4 وطلب 10 أيام", use_container_width=True):
            clicked_query = q6

    st.markdown("---")

    # Display chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["user_query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            # Render Tool Result if available
            if item.get("tool_result") and item["tool_result"].get("success"):
                t_data = item["tool_result"]["data"]
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
    user_input = st.chat_input("اكتب استفسارك هنا حول سياسات الموارد البشرية أو حساب الرصيد...")
    active_prompt = clicked_query or user_input

    if active_prompt:
        # Display user question
        with st.chat_message("user"):
            st.write(active_prompt)

        # Process with Assistant
        with st.chat_message("assistant"):
            with st.spinner("جاري فحص السياسات وتشغيل الأدوات المعتمدة..."):
                assistant = get_assistant()
                response = assistant.process_query(active_prompt)

            st.markdown(response.answer)

            # Tool Result Card
            if response.tool_result and response.tool_result.get("success"):
                t_data = response.tool_result["data"]
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
        )

        requested_input = st.number_input(
            "عدد أيام الإجازة السنوية المطلوبة (بالأيام)",
            min_value=0.5,
            max_value=60.0,
            value=6.0,
            step=0.5,
            help="عدد الأيام المراد التمتع بها (يجب أن يكون أكبر من 0).",
        )

        calc_submitted = st.button("🧮 احسب الرصيد المتبقي والأهلية", type="primary", use_container_width=True)

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
# TAB 3: EVALUATION DASHBOARD
# ==========================================
with tab_evaluation:
    st.markdown("#### 📊 لوحة تقييم الجودة والاختبارات الآلية (12 حالة فحص)")
    st.markdown(
        "تقييم حي وحقيقي يغطي: استرجاع السياسات الخمس، دقة حساب الرصيد، "
        "التحقق الصارم للمدخلات، ورفض العمليات غير المصرح بها، وتحديد الأسئلة الخارجة عن النطاق."
    )

    col_btn_eval, col_metric_eval = st.columns([1, 3])
    with col_btn_eval:
        run_eval_clicked = st.button("🚀 تشغيل جميع الاختبارات الـ 12 آلياً", type="primary", use_container_width=True)

    if run_eval_clicked or st.session_state.eval_results is None:
        with st.spinner("جاري تنفيذ الاختبارات الـ 12 آلياً وقياس الأزمنة الفعلية..."):
            suite = EvaluationSuite()
            st.session_state.eval_results = suite.run_all()

    results = st.session_state.eval_results
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    pass_rate = round((passed_tests / total_tests) * 100, 1)
    total_time = round(sum(r.execution_time_ms for r in results), 2)

    # Metrics Summary Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("إجمالي الاختبارات", f"{total_tests}")
    m2.metric("الاختبارات الناجحة", f"{passed_tests}", delta=f"{passed_tests} ناجح")
    m3.metric("الاختبارات المتعثرة", f"{failed_tests}", delta=f"-{failed_tests}" if failed_tests > 0 else "0")
    m4.metric("نسبة النجاح", f"{pass_rate}%")
    m5.metric("زمن التنفيذ الكلي", f"{total_time} ms")

    st.markdown("---")
    st.markdown("##### 📋 النتائج التفصيلية لكل اختبار:")

    for r in results:
        status_badge = "✅ ناجح (PASSED)" if r.passed else "❌ متعثر (FAILED)"
        card_border = "#86EFAC" if r.passed else "#FCA5A5"
        card_bg = "#FFFFFF"

        with st.expander(f"اختبار #{r.test_id}: {r.name_ar} — {status_badge} ({r.execution_time_ms} ms)", expanded=not r.passed):
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:8px; padding:12px; margin-bottom:8px;">
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
