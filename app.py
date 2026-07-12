"""
JobMatch AI — Main Streamlit Application
AI-powered CV Review & Job Recommendation App

Wizard-style UI with 5 sequential steps:
A. Input CV → B. Lowongan Kerja → C. Review CV → D. Konsultasi Karir → E. Mock Interview
"""

import streamlit as st
import urllib.parse
import re
from pathlib import Path

import config
from cv_processor import extract_cv_text, get_file_info, validate_cv_file
from database import DatabaseManager
from vector_store import VectorStoreManager

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch AI — CV Review & Job Recommendations",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS ─────────────────────────────────────────────
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────
STEPS = [
    {"key": "1", "label": "Input CV", "emoji": "📄", "type": "main"},
    {"key": "2", "label": "Lowongan Kerja", "emoji": "💼", "type": "main"},
    {"key": "3", "label": "Review CV", "emoji": "✍️", "type": "tool"},
    {"key": "4", "label": "Konsultasi Karir", "emoji": "💬", "type": "tool"},
    {"key": "5", "label": "Mock Interview", "emoji": "🎤", "type": "tool"},
]

defaults = {
    "current_step": 0,
    "cv_uploaded": False,
    "cv_text": "",
    "cv_filename": "",
    "cv_file_info": {},
    "cv_bytes": None,
    "job_matches": [],
    "ai_summary": None,
    "selected_job": None,  # posisi target yang dipilih di Step B
    "cv_feedback": None,
    "ats_cv_text": None,
    "career_chat_history": [],
    "interview_history": [],
    "interview_job": None,
    "interview_started": False,
    "bypass_login": False,
    "internet_jobs": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─── Authentication Check ─────────────────────────────────
is_auth_configured = False
try:
    if "auth" in st.secrets and "google" in st.secrets.auth:
        client_id = st.secrets.auth.google.get("client_id", "")
        if client_id and client_id != "MASUKKAN_CLIENT_ID_GOOGLE_DISINI":
            is_auth_configured = True
except Exception:
    pass

if is_auth_configured:
    if not st.user.is_logged_in:
        st.markdown("""
            <div class="hero-container animate-fade-in" style="max-width: 600px; margin: 80px auto; text-align: center; padding: 40px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 20px; box-shadow: var(--shadow-glow);">
                <div style="font-size: 3rem; margin-bottom: 20px;">🎯</div>
                <h1 class="hero-title" style="font-size: 2.2rem; margin-bottom: 15px;">Welcome to JobMatch AI</h1>
                <p class="hero-subtitle" style="font-size: 1rem; margin-bottom: 30px; color: var(--text-secondary);">
                    Silakan masuk menggunakan Google Account Anda untuk mulai menganalisis CV dan mencari kecocokan lowongan kerja.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔑 Log in dengan Google", type="primary", use_container_width=True):
                st.login("google")
        st.stop()
else:
    if not st.session_state.bypass_login:
        st.markdown("""
            <div class="hero-container animate-fade-in" style="max-width: 650px; margin: 60px auto; padding: 40px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 20px; box-shadow: var(--shadow-glow);">
                <div style="font-size: 3rem; margin-bottom: 20px; text-align: center;">🎯</div>
                <h1 class="hero-title" style="font-size: 2.2rem; margin-bottom: 15px; text-align: center;">JobMatch AI Setup</h1>
                <div style="font-size: 0.95rem; line-height: 1.6; color: var(--text-secondary); margin-bottom: 30px;">
                    <p style="color: var(--accent-amber); font-weight: 600; margin-bottom: 12px; text-align: center;">
                        ⚠️ Google Authentication Belum Dikonfigurasi
                    </p>
                    <p>Untuk mengaktifkan login Google, silakan lakukan langkah berikut:</p>
                    <ol style="margin-left: 20px; margin-top: 8px;">
                        <li>Buka Google Cloud Console dan buat OAuth Client ID.</li>
                        <li>Set <strong>Authorized Redirect URI</strong> ke: <code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: var(--text-primary);">http://localhost:8501/oauth2callback</code></li>
                        <li>Buka file <code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: var(--text-primary);">.streamlit/secrets.toml</code> di folder project ini.</li>
                        <li>Masukkan <strong>Client ID</strong> dan <strong>Client Secret</strong> Anda ke dalamnya.</li>
                    </ol>
                </div>
                <div style="text-align: center;">
                    <form method="get">
                        <button name="bypass" value="true" style="background: transparent; border: 1px solid var(--border-color); color: var(--text-secondary); padding: 8px 16px; border-radius: 8px; cursor: pointer;">
                            Bypass Login (Lokal Mode)
                        </button>
                    </form>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Handle bypass parameter from url or button
        query_params = st.query_params
        if query_params.get("bypass") == "true" or st.button("Masuk Mode Lokal (Bypass)", type="secondary", use_container_width=True):
            st.session_state.bypass_login = True
            st.rerun()
        st.stop()


# ─── Helper Functions ─────────────────────────────────────
def go_to_step(step_idx: int):
    """Navigate to a specific step."""
    if step_idx == 0:
        st.session_state.current_step = step_idx
    elif step_idx == 1 and st.session_state.cv_uploaded:
        st.session_state.current_step = step_idx
    elif step_idx >= 2 and st.session_state.cv_uploaded and st.session_state.selected_job:
        st.session_state.current_step = step_idx


def next_step():
    """Go to the next step."""
    if st.session_state.current_step < len(STEPS) - 1:
        go_to_step(st.session_state.current_step + 1)


def prev_step():
    """Go to the previous step."""
    if st.session_state.current_step > 0:
        go_to_step(st.session_state.current_step - 1)


def render_match_badge(score: float) -> str:
    """Generate HTML for a match score badge."""
    if score >= 70:
        cls = "high"
    elif score >= 50:
        cls = "medium"
    else:
        cls = "low"
    return f'<span class="match-badge {cls}">🎯 {score}% Match</span>'


def format_ai_summary(text: str) -> str:
    """Format AI summary headings to be centered, bold, and larger, while avoiding markdown parsing bugs."""
    if not text:
        return ""
    
    # Extract candidate name from title or fallback
    candidate_name = "Kandidat"
    for line in text.split("\n"):
        stripped = line.strip()
        if "analisis" in stripped.lower() and "cv" in stripped.lower():
            # Extract name by removing prefixes/suffixes
            name = re.sub(r'^(?:#+\s+)?(?:Analisis\s+Profil\s+CV|Analisis\s+CV)\s+', '', stripped, flags=re.IGNORECASE)
            name = name.rstrip(":")
            if name:
                candidate_name = name
                break

    lines = text.strip().split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        lower_line = stripped.lower()
        
        # 1. Main Title
        if "analisis" in lower_line and "cv" in lower_line:
            cleaned_lines.append(f'<h2 style="text-align: center; font-weight: 800; font-size: 1.45rem; color: var(--accent-blue); margin-bottom: 20px; border-bottom: 2px solid var(--border-color); padding-bottom: 12px;">📊 CV Analysis: {candidate_name}</h2>')
        
        # 2. Skill Utama
        elif "skill" in lower_line and "utama" in lower_line:
            cleaned_lines.append('<h3 style="text-align: center; font-weight: 800; font-size: 1.2rem; margin-top: 24px; margin-bottom: 12px; color: var(--accent-blue);">1. Skill Utama</h3>')
        
        # 3. Analisis Kesesuaian
        elif "analisis" in lower_line and ("kecocokan" in lower_line or "kesesuaian" in lower_line):
            cleaned_lines.append('<h3 style="text-align: center; font-weight: 800; font-size: 1.2rem; margin-top: 28px; margin-bottom: 12px; color: var(--accent-blue);">2. Analisis Kesesuaian Lowongan</h3>')
        
        # 4. Rekomendasi
        elif "rekomendasi" in lower_line and "cocok" in lower_line:
            cleaned_lines.append('<h3 style="text-align: center; font-weight: 800; font-size: 1.2rem; margin-top: 28px; margin-bottom: 12px; color: var(--accent-blue);">3. Rekomendasi Lowongan Paling Cocok</h3>')
        
        else:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)


def _source_badge_html(source: str) -> str:
    """Generate HTML badge for job source."""
    badge_map = {
        "Dataset": ("📊", "#f59e0b", "rgba(245, 158, 11, 0.12)"),
        "LinkedIn": ("🔗", "#0077b5", "rgba(0, 119, 181, 0.12)"),
        "JobStreet": ("🏢", "#5843be", "rgba(88, 67, 190, 0.12)"),
        "Google Jobs": ("🔍", "#4285f4", "rgba(66, 133, 244, 0.12)"),
    }
    emoji, color, bg = badge_map.get(source, ("🌐", "#94a3b8", "rgba(148, 163, 184, 0.12)"))
    return (
        f'<span style="display:inline-flex; align-items:center; gap:4px; '
        f'padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; '
        f'color:{color}; background:{bg}; border:1px solid {color}30;">'
        f'{emoji} {source}</span>'
    )


def render_job_card(job: dict, show_score: bool = True, is_selected: bool = False, source: str = "") -> str:
    """Generate HTML for a job listing card."""
    meta = job.get("metadata", job)
    title = meta.get("job_title", "Unknown Position")
    company = meta.get("company_name", "Unknown Company")
    location = meta.get("location", "N/A")
    work_type = meta.get("work_type", "N/A")
    salary = meta.get("salary", meta.get("salary_raw", "Tidak disebutkan"))
    if salary == "None" or not salary:
        salary = "Tidak disebutkan"
    score = job.get("similarity_score", 0)

    right_badges = ""
    if show_score and score > 0:
        right_badges += render_match_badge(score)
    if source:
        right_badges += f" {_source_badge_html(source)}"

    html_str = f"""
    <div style="width:100%; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="job-title" style="font-size:1.15rem; font-weight:700; margin-bottom:6px;">{title}</div>
                <div class="job-company" style="font-size:0.95rem; color:var(--accent-blue); font-weight:500; margin-bottom:10px;">🏢 {company}</div>
            </div>
            <div style="display:flex; gap:6px; align-items:center; flex-shrink:0;">{right_badges}</div>
        </div>
        <div class="job-meta" style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
            <span class="job-tag location">📍 {location}</span>
            <span class="job-tag work-type">💼 {work_type}</span>
            <span class="job-tag salary">💰 {salary}</span>
        </div>
    </div>
    """
    # Flatten the HTML to prevent markdown parser from generating a code block
    return html_str.replace("\n", " ").replace("  ", " ").strip()


# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎯 JobMatch AI")
    st.markdown("---")

    # Helper to calculate step status
    def get_step_status(idx):
        is_active = idx == st.session_state.current_step
        if idx == 0:
            is_completed = st.session_state.cv_uploaded
        elif idx == 1:
            is_completed = st.session_state.selected_job is not None
        elif idx == 2:
            is_completed = st.session_state.cv_feedback is not None or st.session_state.ats_cv_text is not None
        elif idx == 3:
            is_completed = len(st.session_state.career_chat_history) > 0
        elif idx == 4:
            is_completed = st.session_state.interview_started
            
        is_locked = (idx > 0 and not st.session_state.cv_uploaded) or (
            idx >= 2 and not st.session_state.selected_job
        )
        
        prefix = "👉 " if is_active else ("✅ " if is_completed else "🔒 " if is_locked else "")
        return is_active, is_completed, is_locked, prefix

    # 1. Input CV
    is_active, is_completed, is_locked, prefix = get_step_status(0)
    if st.button(
        f"{prefix}📄 1. Input CV",
        key="nav_0",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        disabled=is_locked
    ):
        go_to_step(0)
        st.rerun()

    # Arrow Down Connector
    st.markdown(
        """<div style='text-align:center; color:rgba(255,255,255,0.25); font-size:1.3rem; margin:-6px 0 -8px 0;'>
            ↓
        </div>""",
        unsafe_allow_html=True
    )

    # 2. Lowongan Kerja
    is_active, is_completed, is_locked, prefix = get_step_status(1)
    if st.button(
        f"{prefix}💼 2. Lowongan Kerja",
        key="nav_1",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        disabled=is_locked
    ):
        go_to_step(1)
        st.rerun()

    # Beautiful SVG Branching Diagram
    st.markdown(
        """<svg width="100%" height="28" viewBox="0 0 100 28" style="display:block; margin: 2px auto;">
          <line x1="50" y1="0" x2="50" y2="12" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
          <line x1="16.6" y1="12" x2="83.3" y2="12" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
          <line x1="16.6" y1="12" x2="16.6" y2="28" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
          <line x1="50" y1="12" x2="50" y2="28" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
          <line x1="83.3" y1="12" x2="83.3" y2="28" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
          <polygon points="16.6,28 13.6,23 19.6,23" fill="rgba(255,255,255,0.25)"/>
          <polygon points="50,28 47,23 53,23" fill="rgba(255,255,255,0.25)"/>
          <polygon points="83.3,28 80.3,23 86.3,23" fill="rgba(255,255,255,0.25)"/>
        </svg>""",
        unsafe_allow_html=True
    )

    # 3 Parallel Columns for optional AI tools
    col1, col2, col3 = st.columns(3)
    
    with col1:
        is_active, is_completed, is_locked, prefix = get_step_status(2)
        label = f"{prefix}✍️\nReview"
        if st.button(
            label,
            key="nav_2",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_locked,
            help="Review & Saran CV"
        ):
            go_to_step(2)
            st.rerun()
            
    with col2:
        is_active, is_completed, is_locked, prefix = get_step_status(3)
        label = f"{prefix}💬\nKonsul"
        if st.button(
            label,
            key="nav_3",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_locked,
            help="Konsultasi Karir"
        ):
            go_to_step(3)
            st.rerun()
            
    with col3:
        is_active, is_completed, is_locked, prefix = get_step_status(4)
        label = f"{prefix}🎤\nMock"
        if st.button(
            label,
            key="nav_4",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_locked,
            help="Mock Interview"
        ):
            go_to_step(4)
            st.rerun()

    # API Status
    st.markdown("---")
    if config.is_openai_configured():
        st.success("✅ OpenAI API Connected", icon="🔑")
    else:
        st.warning("⚠️ OpenAI API key belum diatur", icon="🔑")
        st.caption("Tambahkan di file `.env`")

    # N8N Status
    _use_n8n = config.USE_N8N or config._get_config("USE_N8N", "false").lower() == "true"
    if _use_n8n:
        if config.is_n8n_configured():
            st.success("✅ N8N Connected", icon="🔗")
        else:
            st.warning("⚠️ N8N URL belum diatur", icon="🔗")
    else:
        st.info("💻 Mode: Local (tanpa N8N)", icon="🏠")
    # User Profile & Logout
    if st.user.is_logged_in:
        st.markdown("---")
        user_name = st.user.get("name", st.user.get("email", "User"))
        user_email = st.user.get("email", "")
        st.markdown(
            f"""<div style="background-color: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 10px;">
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--text-primary);">👤 {user_name}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">{user_email}</div>
            </div>""",
            unsafe_allow_html=True
        )
        if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
            st.logout()
    elif st.session_state.bypass_login:
        st.markdown("---")
        st.info("🛠️ Mode: Developer Bypass")
        if st.button("🚪 Kembali ke Login", use_container_width=True, type="secondary"):
            st.session_state.bypass_login = False
            st.rerun()

# ═══════════════════════════════════════════════════════════
# STEP A: INPUT CV
# ═══════════════════════════════════════════════════════════
if st.session_state.current_step == 0:
    st.markdown(
        """<div class="hero-container animate-fade-in">
            <div class="hero-title">📄 Upload CV Kamu</div>
            <div class="hero-subtitle">
                Upload CV dalam format PDF atau Word untuk memulai analisis AI. 
                Kami akan mencocokkan profil kamu dengan ratusan lowongan pekerjaan.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drag & drop CV kamu di sini",
            type=["pdf", "docx", "doc"],
            help="Format yang didukung: PDF, DOCX. Maksimum 100MB.",
            key="cv_uploader",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()

            # Validate
            is_valid, error_msg = validate_cv_file(file_bytes, uploaded_file.name)

            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Extract text
                with st.spinner("📖 Membaca CV kamu..."):
                    try:
                        cv_text = extract_cv_text(file_bytes, uploaded_file.name)
                        file_info = get_file_info(file_bytes, uploaded_file.name)

                        # Save to session state
                        st.session_state.cv_uploaded = True
                        st.session_state.cv_text = cv_text
                        st.session_state.cv_filename = uploaded_file.name
                        st.session_state.cv_file_info = file_info
                        st.session_state.cv_bytes = file_bytes

                        st.success("✅ CV berhasil di-upload dan dibaca!")

                    except Exception as e:
                        st.error(f"❌ Gagal membaca CV: {str(e)}")

    with col2:
        st.markdown(
            """<div class="glass-card">
                <h4 style="color:var(--accent-blue);">📋 Panduan</h4>
                <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6;">
                    <strong>Format:</strong> PDF atau Word<br>
                    <strong>Max Size:</strong> 100MB<br>
                    <strong>Tips:</strong> Pastikan CV kamu berisi informasi yang lengkap tentang pengalaman, skill, dan pendidikan.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Show CV preview if uploaded
    if st.session_state.cv_uploaded:
        st.markdown("---")
        st.markdown("### 📋 Preview CV")

        info = st.session_state.cv_file_info
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-number">{info.get('format', 'N/A')}</div>
                    <div class="stat-label">Format</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-number">{info.get('size_mb', 0)}</div>
                    <div class="stat-label">MB</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_c:
            pages = info.get("pages", info.get("paragraphs", "—"))
            label = "Halaman" if "pages" in info else "Paragraf"
            st.markdown(
                f"""<div class="stat-card">
                    <div class="stat-number">{pages}</div>
                    <div class="stat-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("📄 Lihat Isi CV (Text)", expanded=False):
            st.text_area(
                "CV Content",
                st.session_state.cv_text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

        # Next button
        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 1])
        with col_r:
            if st.button("Lihat Rekomendasi Kerja →", type="primary", use_container_width=True):
                next_step()
                st.rerun()


# ═══════════════════════════════════════════════════════════
# STEP B: LOWONGAN KERJA (UNIFIED VIEW)
# ═══════════════════════════════════════════════════════════
elif st.session_state.current_step == 1:
    st.markdown(
        """<div class="hero-container animate-fade-in">
            <div class="hero-title">💼 Rekomendasi Lowongan Kerja</div>
            <div class="hero-subtitle">
                AI mencocokkan CV kamu dengan database lowongan pekerjaan di Indonesia dan memberikan saran posisi dari berbagai platform.<br>
                <strong style="color:var(--accent-emerald);">Pilih satu posisi target</strong> untuk melanjutkan ke Review CV, Konsultasi Karir, dan Mock Interview.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Run Dataset matching if not done yet ──
    if not st.session_state.job_matches:
        with st.spinner("🤖 AI sedang mencocokkan CV kamu dengan database lowongan..."):
            try:
                from agents.rag_agent import match_cv_to_jobs
                result = match_cv_to_jobs(st.session_state.cv_text, top_k=config.TOP_K_RESULTS)
                st.session_state.job_matches = result.get("matches", [])
                st.session_state.ai_summary = result.get("ai_summary")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ── Run AI internet suggestions if not done yet ──
    if not st.session_state.internet_jobs:
        with st.spinner("🌐 AI sedang mencari saran lowongan dari berbagai platform..."):
            try:
                from agents.web_job_agent import generate_job_suggestions
                st.session_state.internet_jobs = generate_job_suggestions(st.session_state.cv_text)
            except Exception:
                st.session_state.internet_jobs = []

    # ── Show AI summary if available ──
    if st.session_state.ai_summary:
        with st.expander("🤖 Analisis AI", expanded=True):
            formatted_summary = format_ai_summary(st.session_state.ai_summary)
            st.markdown(formatted_summary, unsafe_allow_html=True)

    # ── Show selected position banner ──
    if st.session_state.selected_job:
        sel = st.session_state.selected_job
        sel_salary = sel.get("salary", sel.get("salary_raw", "Tidak disebutkan"))
        if sel_salary == "None" or not sel_salary:
            sel_salary = "Tidak disebutkan"
        sel_source = sel.get("source", "")
        source_badge = f" {_source_badge_html(sel_source)}" if sel_source else ""
        st.markdown(
            f"""<div class="glass-card" style="border-left: 4px solid var(--accent-emerald); margin-bottom: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="color:var(--accent-emerald); margin:0;">🎯 Posisi Target Terpilih {source_badge}</h4>
                        <p style="font-size:1.1rem; font-weight:600; color:var(--text-color, #e2e8f0); margin:4px 0 2px 0;">
                            {sel.get('job_title', 'N/A')} — {sel.get('company_name', 'N/A')}
                        </p>
                        <span class="job-tag location">📍 {sel.get('location', 'N/A')}</span>
                        <span class="job-tag work-type">💼 {sel.get('work_type', 'N/A')}</span>
                        <span class="job-tag salary">💰 {sel_salary}</span>
                    </div>
                </div>
                <p style="font-size:0.8rem; color:var(--text-secondary); margin-top:8px;">
                    ✅ Step C (Review CV), D (Konsultasi Karir), dan E (Mock Interview) akan diarahkan untuk posisi ini.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Instruction banner if no position selected yet ──
    if not st.session_state.selected_job:
        st.markdown(
            """<div class="glass-card" style="border-left: 4px solid var(--accent-amber); margin-bottom: 1rem;">
                <h4 style="color:var(--accent-amber); margin:0;">👇 Pilih Posisi Target Kamu</h4>
                <p style="font-size:0.88rem; color:var(--text-secondary); margin:4px 0 0 0;">
                    Klik tombol <strong>"🎯 Targetkan Posisi Ini"</strong> pada lowongan yang kamu minati.
                    Setelah memilih, Review CV, Konsultasi Karir, dan Mock Interview akan disesuaikan untuk posisi tersebut.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Helper function to render a selectable job card ──
    def _render_selectable_card(job_data: dict, card_key: str, source: str, description: str = ""):
        """Render a job card with select button inside a bordered container."""
        meta = job_data if source != "Dataset" else job_data.get("metadata", {})
        job_title = meta.get("job_title", "")
        company_name = meta.get("company_name", "")

        is_selected = (
            st.session_state.selected_job is not None
            and st.session_state.selected_job.get("job_title") == job_title
            and st.session_state.selected_job.get("company_name") == company_name
        )

        with st.container(border=True):
            if is_selected:
                st.markdown(
                    '<div style="background-color: rgba(16, 185, 129, 0.08); '
                    'border-left: 4px solid var(--accent-emerald); '
                    'padding: 8px 12px; border-radius: 6px; font-size: 0.88rem; '
                    'font-weight: 700; color: var(--accent-emerald); margin-bottom: 12px;">'
                    '🎯 Posisi Target Aktif Anda</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(render_job_card(job_data, is_selected=is_selected, source=source), unsafe_allow_html=True)

            col_sel, col_empty = st.columns([1, 1])
            with col_sel:
                if is_selected:
                    st.success("✅ Terpilih sebagai target")
                else:
                    if st.button("🎯 Targetkan Posisi Ini", key=card_key, type="primary", use_container_width=True):
                        desc = description or job_data.get("document", "") or meta.get("description", "")
                        st.session_state.selected_job = {
                            "job_title": job_title,
                            "company_name": company_name,
                            "location": meta.get("location", ""),
                            "work_type": meta.get("work_type", ""),
                            "salary": meta.get("salary", "None"),
                            "job_description": desc,
                            "source": source,
                        }
                        st.session_state.interview_job = {
                            "job_title": job_title,
                            "company_name": company_name,
                            "job_description": desc,
                        }
                        st.session_state.cv_feedback = None
                        st.session_state.ats_cv_text = None
                        st.session_state.career_chat_history = []
                        st.rerun()

            with st.expander("📋 Lihat Detail Lowongan", expanded=False):
                detail = description or job_data.get("document", "") or meta.get("description", "Tidak ada deskripsi.")
                st.markdown(detail)

    # ═══════════════════════════════════════════════════════
    # SECTION 1: DARI DATABASE (DATASET)
    # ═══════════════════════════════════════════════════════
    if st.session_state.job_matches:
        dataset_count = len(st.session_state.job_matches)
        internet_count = len(st.session_state.internet_jobs)
        total_count = dataset_count + internet_count
        st.markdown(f"### 📊 Dari Database — {dataset_count} Lowongan Cocok")

        # Filters for dataset
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_score = st.slider("Minimum Match Score", 0, 100, 0, step=5)
        with col_f2:
            sort_order = st.selectbox("Urutkan", ["Tertinggi", "Terendah"])

        filtered = [j for j in st.session_state.job_matches if j.get("similarity_score", 0) >= min_score]
        filtered.sort(key=lambda x: x.get("similarity_score", 0), reverse=(sort_order == "Tertinggi"))

        for idx, job in enumerate(filtered):
            _render_selectable_card(job, f"select_ds_{idx}", source="Dataset")
    else:
        if st.session_state.cv_uploaded:
            from vector_store import VectorStoreManager
            vs_count = 0
            try:
                vs_count = VectorStoreManager().get_collection_count()
            except Exception:
                pass
            cv_len = len(st.session_state.cv_text) if st.session_state.cv_text else 0
            st.info(f"🔍 Belum ada hasil dari database. (Panjang CV: {cv_len} karakter, Jumlah lowongan di DB: {vs_count}).")
            if st.button("🔄 Cari Ulang di Database"):
                st.session_state.job_matches = []
                st.rerun()

    # ═══════════════════════════════════════════════════════
    # SECTION 2: SARAN AI DARI BERBAGAI PLATFORM
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    if st.session_state.internet_jobs:
        st.markdown(f"### 🌐 Saran AI dari Berbagai Platform — {len(st.session_state.internet_jobs)} Rekomendasi")
        st.caption("💡 Lowongan di bawah ini adalah rekomendasi AI berdasarkan analisis CV kamu. Klik link platform untuk melihat lowongan serupa yang sebenarnya.")

        # Group by source
        for source_name in ["LinkedIn", "JobStreet", "Google Jobs"]:
            source_jobs = [j for j in st.session_state.internet_jobs if j.get("source") == source_name]
            if source_jobs:
                for idx, ijob in enumerate(source_jobs):
                    _render_selectable_card(ijob, f"select_web_{source_name}_{idx}", source=source_name, description=ijob.get("description", ""))

        # Quick links to real platforms
        search_keyword = ""
        if st.session_state.job_matches:
            search_keyword = st.session_state.job_matches[0].get("metadata", {}).get("job_title", "")
        elif st.session_state.internet_jobs:
            search_keyword = st.session_state.internet_jobs[0].get("job_title", "")
        if search_keyword:
            encoded = urllib.parse.quote(search_keyword)
            st.markdown("#### 🔗 Cari Langsung di Platform")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.link_button("🔗 Buka LinkedIn", f"https://www.linkedin.com/jobs/search/?keywords={encoded}&location=Indonesia", use_container_width=True)
            with col2:
                st.link_button("🏢 Buka JobStreet", f"https://www.jobstreet.co.id/id/job-search/{encoded}-jobs/", use_container_width=True)
            with col3:
                st.link_button("🔍 Buka Google Jobs", f"https://www.google.com/search?q={encoded}+jobs+Indonesia&ibp=htl;jobs", use_container_width=True)
    else:
        st.info("🌐 Tidak ada saran lowongan dari internet. Pastikan OpenAI API key sudah diatur.")
        if st.button("🔄 Cari Ulang Saran Internet"):
            st.session_state.internet_jobs = []
            st.rerun()

    # Navigation buttons
    st.markdown("---")
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if st.button("← Kembali", use_container_width=True):
            prev_step()
            st.rerun()
    with col_r:
        if st.session_state.selected_job:
            if st.button("Review CV →", type="primary", use_container_width=True):
                next_step()
                st.rerun()
        else:
            st.button("Review CV →", use_container_width=True, disabled=True, help="Pilih posisi target dulu di atas")


# ═══════════════════════════════════════════════════════════
# STEP C: REVIEW CV
# ═══════════════════════════════════════════════════════════
elif st.session_state.current_step == 2:
    # Build subtitle based on selected job
    _cv_subtitle = "AI akan menganalisis CV kamu dan memberikan feedback untuk meningkatkan kualitasnya"
    if st.session_state.selected_job:
        _sel_title = st.session_state.selected_job.get('job_title', '')
        _sel_company = st.session_state.selected_job.get('company_name', '')
        _cv_subtitle = f"AI akan menganalisis CV kamu secara spesifik untuk posisi <strong>{_sel_title}</strong> di <strong>{_sel_company}</strong>"

    st.markdown(
        f"""<div class="hero-container animate-fade-in">
            <div class="hero-title">✍️ Review & Saran CV</div>
            <div class="hero-subtitle">
                {_cv_subtitle}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Guard: redirect if no target selected
    if not st.session_state.selected_job:
        st.warning(
            "⚠️ Kamu belum memilih posisi target. Kembali ke Step B untuk memilih posisi yang kamu minati.",
            icon="⚠️",
        )
        if st.button("← Kembali ke Lowongan Kerja", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    elif not config.is_openai_configured():
        st.warning(
            "⚠️ Fitur ini membutuhkan OpenAI API key. Tambahkan `OPENAI_API_KEY` di file `.env`",
            icon="🔑",
        )
    else:
        tab1, tab2 = st.tabs(["📊 Feedback & Saran", "📝 Generate CV ATS"])

        # ── Tab 1: CV Feedback ──
        with tab1:
            if st.session_state.cv_feedback is None:
                if st.button("🤖 Analisis CV Saya", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI sedang menganalisis CV kamu..."):
                        from agents.cv_analyzer_agent import review_cv
                        result = review_cv(st.session_state.cv_text, target_job=st.session_state.selected_job)
                        if result["available"] and result["feedback"]:
                            st.session_state.cv_feedback = result["feedback"]
                            st.rerun()
                        else:
                            st.error("❌ Gagal menganalisis CV.")
            else:
                st.markdown(st.session_state.cv_feedback)

                if st.button("🔄 Analisis Ulang"):
                    st.session_state.cv_feedback = None
                    st.rerun()

        # ── Tab 2: ATS CV Generation ──
        with tab2:
            st.markdown(
                """<div class="glass-card">
                    <h4 style="color:var(--accent-emerald);">📝 Generate CV ATS-Friendly</h4>
                    <p style="font-size:0.9rem; color:var(--text-secondary);">
                        AI akan membuat versi CV kamu yang dioptimalkan untuk Applicant Tracking System (ATS).
                        Kamu bisa download hasilnya dalam format Word atau PDF.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

            if st.session_state.ats_cv_text is None:
                if st.button("✨ Generate CV ATS", type="primary", use_container_width=True):
                    with st.spinner("✨ AI sedang membuat CV ATS-friendly..."):
                        from agents.cv_analyzer_agent import generate_ats_cv
                        result = generate_ats_cv(st.session_state.cv_text, target_job=st.session_state.selected_job)
                        if result["available"] and result["ats_text"]:
                            st.session_state.ats_cv_text = result["ats_text"]
                            st.rerun()
                        else:
                            st.error("❌ Gagal membuat CV ATS.")
            else:
                st.markdown("### 📄 Preview CV ATS")
                # Clean up markdown code block wrappers if present
                clean_ats_text = st.session_state.ats_cv_text.strip()
                if clean_ats_text.startswith("```"):
                    lines = clean_ats_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_ats_text = "\n".join(lines).strip()

                import markdown
                html_cv = markdown.markdown(clean_ats_text)
                st.markdown(
                    f'<div class="cv-paper">{html_cv}</div>',
                    unsafe_allow_html=True
                )

                # Download buttons
                st.markdown("### 📥 Download CV ATS")
                col1, col2 = st.columns(2)

                with col1:
                    try:
                        from agents.cv_analyzer_agent import export_cv_to_docx
                        docx_bytes = export_cv_to_docx(clean_ats_text)
                        st.download_button(
                            "📄 Download Word (.docx)",
                            data=docx_bytes,
                            file_name="CV_ATS_Optimized.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Error generating DOCX: {e}")

                with col2:
                    try:
                        from agents.cv_analyzer_agent import export_cv_to_pdf
                        pdf_bytes = export_cv_to_pdf(clean_ats_text)
                        st.download_button(
                            "📑 Download PDF (.pdf)",
                            data=pdf_bytes,
                            file_name="CV_ATS_Optimized.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")

                if st.button("🔄 Generate Ulang"):
                    st.session_state.ats_cv_text = None
                    st.rerun()

    # Navigation
    st.markdown("---")
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if st.button("← Kembali", use_container_width=True):
            prev_step()
            st.rerun()
    with col_r:
        if st.button("Konsultasi Karir →", type="primary", use_container_width=True):
            next_step()
            st.rerun()


# ═══════════════════════════════════════════════════════════
# STEP D: KONSULTASI KARIR
# ═══════════════════════════════════════════════════════════
elif st.session_state.current_step == 3:
    # Build subtitle based on selected job
    _career_subtitle = "Diskusikan cita-cita dan tujuan karir kamu dengan AI Career Consultant"
    if st.session_state.selected_job:
        _sel_title = st.session_state.selected_job.get('job_title', '')
        _career_subtitle = f"Diskusikan strategi karir kamu untuk mengejar posisi <strong>{_sel_title}</strong> dengan AI Career Consultant"

    st.markdown(
        f"""<div class="hero-container animate-fade-in">
            <div class="hero-title">💬 Konsultasi Karir</div>
            <div class="hero-subtitle">
                {_career_subtitle}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not config.is_openai_configured():
        st.warning(
            "⚠️ Fitur ini membutuhkan OpenAI API key. Tambahkan `OPENAI_API_KEY` di file `.env`",
            icon="🔑",
        )
    else:
        # Display chat history
        for msg in st.session_state.career_chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Welcome message if no history
        if not st.session_state.career_chat_history:
            if st.session_state.selected_job:
                _sel = st.session_state.selected_job
                welcome_msg = f"""<div class="chat-ai">
                    🤖 Halo! Saya AI Career Consultant kamu. Saya sudah membaca CV kamu.<br><br>
                    Saya lihat kamu tertarik dengan posisi <strong>{_sel.get('job_title', '')}</strong> di <strong>{_sel.get('company_name', '')}</strong>.<br><br>
                    Silakan ceritakan tentang:<br>
                    • 🎯 Kenapa kamu tertarik dengan posisi ini?<br>
                    • 🤔 Apa yang kamu rasakan kurang dari profil kamu saat ini?<br>
                    • 📈 Skill apa yang ingin kamu kembangkan untuk posisi ini?<br>
                    • 💡 Atau apapun tentang karir kamu!
                </div>"""
            else:
                welcome_msg = """<div class="chat-ai">
                    🤖 Halo! Saya AI Career Consultant kamu. Saya sudah membaca CV kamu.<br><br>
                    Silakan ceritakan tentang:<br>
                    • 🎯 Cita-cita atau tujuan karir kamu<br>
                    • 🤔 Keraguan tentang pilihan karir<br>
                    • 📈 Skill yang ingin dikembangkan<br>
                    • 💡 Atau apapun tentang karir kamu!
                </div>"""
            st.markdown(welcome_msg, unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("Ketik pesan kamu di sini...")

        if user_input:
            # Add user message
            st.session_state.career_chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Get AI response
            with st.spinner("🤖 AI sedang berpikir..."):
                from agents.career_agent import get_career_response
                result = get_career_response(
                    cv_text=st.session_state.cv_text,
                    chat_history=st.session_state.career_chat_history[:-1],  # exclude last
                    user_message=user_input,
                    target_job=st.session_state.selected_job,
                )

                if result["available"] and result["response"]:
                    st.session_state.career_chat_history.append(
                        {"role": "assistant", "content": result["response"]}
                    )
                else:
                    st.session_state.career_chat_history.append(
                        {"role": "assistant", "content": "Maaf, terjadi kesalahan. Coba lagi."}
                    )

            st.rerun()

        # Clear chat button
        if st.session_state.career_chat_history:
            if st.button("🗑️ Hapus Riwayat Chat"):
                st.session_state.career_chat_history = []
                st.rerun()

    # Navigation
    st.markdown("---")
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if st.button("← Kembali", use_container_width=True):
            prev_step()
            st.rerun()
    with col_r:
        if st.button("Mock Interview →", type="primary", use_container_width=True):
            next_step()
            st.rerun()


# ═══════════════════════════════════════════════════════════
# STEP E: MOCK INTERVIEW
# ═══════════════════════════════════════════════════════════
elif st.session_state.current_step == 4:
    st.markdown(
        """<div class="hero-container animate-fade-in">
            <div class="hero-title">🎤 Mock Interview</div>
            <div class="hero-subtitle">
                Simulasi interview dengan AI sebagai HR Interviewer. Pilih mode text atau voice.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not config.is_openai_configured():
        st.warning(
            "⚠️ Fitur ini membutuhkan OpenAI API key. Tambahkan `OPENAI_API_KEY` di file `.env`",
            icon="🔑",
        )
    else:
        # Job selection for interview
        if not st.session_state.interview_job:
            # Auto-set from selected_job if available
            if st.session_state.selected_job:
                st.session_state.interview_job = {
                    "job_title": st.session_state.selected_job.get("job_title", ""),
                    "company_name": st.session_state.selected_job.get("company_name", ""),
                    "job_description": st.session_state.selected_job.get("job_description", ""),
                }
                st.rerun()

            st.markdown("### 🎯 Pilih Posisi untuk Interview")

            if st.session_state.job_matches:
                job_options = {}
                for j in st.session_state.job_matches:
                    meta = j.get("metadata", {})
                    label = f"{meta.get('job_title', 'N/A')} — {meta.get('company_name', 'N/A')}"
                    job_options[label] = {
                        "job_title": meta.get("job_title", ""),
                        "company_name": meta.get("company_name", ""),
                        "job_description": j.get("document", ""),
                    }

                selected = st.selectbox("Pilih lowongan:", list(job_options.keys()))
                if st.button("🎬 Mulai Interview", type="primary"):
                    st.session_state.interview_job = job_options[selected]
                    st.rerun()
            else:
                st.info("💡 Upload CV dan lihat rekomendasi dulu (Step A & B) untuk memilih posisi interview.")

                # Manual input option
                st.markdown("**Atau input manual:**")
                manual_title = st.text_input("Job Title", placeholder="contoh: Data Analyst")
                manual_company = st.text_input("Company Name", placeholder="contoh: PT ABC")
                manual_desc = st.text_area("Job Description (opsional)", placeholder="Deskripsi pekerjaan...")

                if manual_title and st.button("🎬 Mulai Interview", type="primary"):
                    st.session_state.interview_job = {
                        "job_title": manual_title,
                        "company_name": manual_company or "Unknown Company",
                        "job_description": manual_desc or "N/A",
                    }
                    st.rerun()

        else:
            # Show interview info
            job = st.session_state.interview_job
            st.markdown(
                f"""<div class="glass-card">
                    <h4 style="color:var(--accent-blue);">🎯 Posisi: {job['job_title']}</h4>
                    <p style="color:var(--text-secondary);">🏢 {job['company_name']}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Mode selection
            mode = st.radio(
                "Mode Interview:",
                ["💬 Text", "🎙️ Voice"],
                horizontal=True,
            )

            # Start interview if not started
            if not st.session_state.interview_started:
                if st.button("🎬 Mulai Interview Sekarang", type="primary"):
                    with st.spinner("🤖 HR sedang mempersiapkan interview..."):
                        from agents.interview_agent import start_interview
                        result = start_interview(st.session_state.cv_text, job)
                        if result["available"] and result["response"]:
                            st.session_state.interview_history = [
                                {"role": "assistant", "content": result["response"]}
                            ]
                            st.session_state.interview_started = True
                            st.rerun()
            else:
                # Display interview conversation
                for msg in st.session_state.interview_history:
                    if msg["role"] == "assistant":
                        st.markdown(
                            f'<div class="chat-ai">🤵 HR: {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )

                        # TTS for voice mode
                        if mode == "🎙️ Voice" and msg == st.session_state.interview_history[-1]:
                            try:
                                from agents.interview_agent import text_to_speech
                                audio_bytes = text_to_speech(msg["content"])
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                            except Exception:
                                pass

                    else:
                        st.markdown(
                            f'<div class="chat-user">🧑 Kamu: {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )

                # Input area
                if mode == "💬 Text":
                    answer = st.chat_input("Ketik jawaban kamu...")
                    if answer:
                        st.session_state.interview_history.append(
                            {"role": "user", "content": answer}
                        )
                        with st.spinner("🤵 HR sedang mengevaluasi jawaban..."):
                            from agents.interview_agent import continue_interview
                            result = continue_interview(
                                st.session_state.cv_text,
                                job,
                                st.session_state.interview_history[:-1],
                                answer,
                            )
                            if result["available"] and result["response"]:
                                st.session_state.interview_history.append(
                                    {"role": "assistant", "content": result["response"]}
                                )
                        st.rerun()

                else:  # Voice mode
                    st.markdown("### 🎙️ Rekam Jawaban")
                    try:
                        from audio_recorder_streamlit import audio_recorder
                        audio_bytes = audio_recorder(
                            text="Klik untuk mulai merekam",
                            recording_color="#f43f5e",
                            neutral_color="#00d4ff",
                            icon_size="2x",
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")
                            if st.button("📤 Kirim Jawaban", type="primary"):
                                with st.spinner("🎧 Transcribing audio..."):
                                    from agents.interview_agent import (
                                        transcribe_audio,
                                        continue_interview,
                                    )
                                    transcribed = transcribe_audio(audio_bytes)
                                    st.info(f"📝 Transcribed: {transcribed}")

                                    st.session_state.interview_history.append(
                                        {"role": "user", "content": transcribed}
                                    )

                                    result = continue_interview(
                                        st.session_state.cv_text,
                                        job,
                                        st.session_state.interview_history[:-1],
                                        transcribed,
                                    )
                                    if result["available"] and result["response"]:
                                        st.session_state.interview_history.append(
                                            {"role": "assistant", "content": result["response"]}
                                        )
                                    st.rerun()
                    except ImportError:
                        st.warning("📦 Package `audio-recorder-streamlit` belum terinstall.")
                        st.code("pip install audio-recorder-streamlit")
                        st.info("Gunakan mode Text untuk sementara.")

            # Reset interview
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Interview"):
                    st.session_state.interview_started = False
                    st.session_state.interview_history = []
                    st.rerun()
            with col2:
                if st.button("🔄 Ganti Posisi"):
                    st.session_state.interview_job = None
                    st.session_state.interview_started = False
                    st.session_state.interview_history = []
                    st.rerun()

    # Navigation
    st.markdown("---")
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        if st.button("← Kembali", use_container_width=True):
            prev_step()
            st.rerun()
    with col_r:
        st.markdown(
            """<div style="text-align:center; padding:10px;">
                <span style="color:var(--accent-emerald); font-weight:600;">
                    🎉 Ini step terakhir!
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
