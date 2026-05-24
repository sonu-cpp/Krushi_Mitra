import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.translations import t
from utils.crop_utils import predict_crop, predict_top_crops, predict_top_crops_merged, get_crop_info
from utils.fertilizer_utils import get_all_crops, get_soil_types_for_crop, recommend_fertilizer


st.set_page_config(page_title="Krushi Mitra", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a4731 0%, #2d6a4f 100%); }
    [data-testid="stSidebar"] * { color: #fefae0 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * { color: #1a4731 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { background: white !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #fefae0 !important; }
    .main-header { background: linear-gradient(135deg, #1a4731 0%, #2d6a4f 60%, #52b788 100%); padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; }
    .main-header h1 { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; margin: 0; color: #e9c46a; }
    .main-header p { font-size: 1.05rem; color: #c8e6c9; margin: 0.4rem 0 0 0; }
    .feature-card { background: white; border: 1px solid #e8f5e9; border-radius: 14px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 12px rgba(26,71,49,0.07); }
    .feature-card .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .feature-card h3 { color: #1a4731; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    .feature-card p { color: #555; font-size: 0.9rem; line-height: 1.5; }
    .result-box { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #52b788; border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1.5rem; }
    .result-box h2 { color: #1a4731; font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0 0 0.3rem 0; text-transform: capitalize; }
    .result-box .confidence { color: #2d6a4f; font-weight: 600; font-size: 0.95rem; }
    .result-box .crop-detail { background: white; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.8rem; font-size: 0.9rem; color: #444; }
    .advice-item { background: #fffbeb; border-left: 4px solid #e9c46a; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.5rem 0; font-size: 0.93rem; color: #333; }
    .advice-good { background: #f0fdf4; border-left: 4px solid #52b788; }
    .section-title { font-family: 'Playfair Display', serif; font-size: 1.7rem; color: #1a4731; font-weight: 700; margin-bottom: 1rem; padding-bottom: 0.4rem; border-bottom: 3px solid #52b788; }
    .stButton > button { background: linear-gradient(135deg, #2d6a4f, #52b788); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; }
    div[data-testid="stMetric"] { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 0.8rem 1rem; }
    .fert-input-card { background: white; border: 1.5px solid #bbf7d0; border-radius: 16px; padding: 2rem 2.2rem 1.5rem 2.2rem; box-shadow: 0 2px 12px rgba(26,71,49,0.07); margin-bottom: 1.5rem; }
    .fert-result-header { background: linear-gradient(135deg, #1a4731 0%, #2d6a4f 60%, #52b788 100%); border-radius: 14px; padding: 1.4rem 2rem; margin: 1.2rem 0; color: white; }
    .fert-result-header h2 { color: #e9c46a; font-family: 'Playfair Display', serif; margin: 0 0 0.2rem 0; font-size: 1.7rem; }
    .fert-result-header p  { color: #c8e6c9; margin: 0; font-size: 0.95rem; }
    .fert-dose-box { background: #fffbeb; border: 2px solid #e9c46a; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0; }
    .fert-dose-box h4 { color: #92400e; margin: 0 0 0.5rem 0; font-size: 1rem; }
    .fert-dose-box p  { color: #333; margin: 0; font-size: 0.95rem; font-weight: 500; }
    .npk-pill { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 99px; font-size: 0.85rem; font-weight: 700; margin: 0 0.3rem 0.3rem 0; }
    .npk-n { background: #dcfce7; color: #166534; }
    .npk-p { background: #dbeafe; color: #1e3a8a; }
    .npk-k { background: #fef9c3; color: #713f12; }
    .npk-ph { background: #fce7f3; color: #831843; }
    .npk-w  { background: #e0f2fe; color: #0c4a6e; }
</style>
""", unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state.lang = "English"

with st.sidebar:
    st.markdown("## 🌾 Krushi Mitra")
    st.markdown("---")
    lang = st.selectbox(t("select_language", st.session_state.lang), ["English", "Hindi", "Odia"],
                        index=["English", "Hindi", "Odia"].index(st.session_state.lang))
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        st.session_state.pop("_active_page", None)
        st.rerun()
    L = lang
    st.markdown("---")
    nav_options = [t("nav_home",L), t("nav_crop",L), t("nav_fertilizer",L), t("nav_disease",L)]
    default_page = st.session_state.get("_active_page", nav_options[0])
    if default_page not in nav_options:
        default_page = nav_options[0]
    page = st.radio("Navigate", nav_options, index=nav_options.index(default_page))
    st.session_state["_active_page"] = page
    st.markdown("---")
    st.markdown("<small style='color:#a5d6a7'>Krushi Mitra v1.0<br>BCA Project · SOA University</small>", unsafe_allow_html=True)

L = st.session_state.lang


# HOME
if page == t("nav_home", L):
    st.markdown(f'<div class="main-header"><h1>🌾 {t("app_title",L)}</h1><p>{t("app_subtitle",L)}</p></div>', unsafe_allow_html=True)
    st.markdown(f"### {t('welcome', L)}")
    st.markdown(t('home_desc', L))
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,icon,feat,desc in [(c1,"🌱",t("feature_crop",L),t("feature_crop_desc",L)),(c2,"🧪",t("feature_fertilizer",L),t("feature_fertilizer_desc",L)),(c3,"🔬",t("feature_disease",L),t("feature_disease_desc",L))]:
        with col:
            st.markdown(f'<div class="feature-card"><div class="icon">{icon}</div><h3>{feat}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📌 **Note:** This system uses AI models trained on agricultural datasets. Results are advisory only. Always consult your local agricultural extension officer for critical decisions.")

# ══════════════════════════════════════════════════════════════════════════════
# CROP RECOMMENDATION  (new district-aware wizard)
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("nav_crop", L):
    from utils.questionnaire_utils import (
        get_states_and_districts,
        get_district_env_data,
        get_district_annual_rainfall,
        get_rainfall_option_labels,
        RAINFALL_FEEDBACK_FACTOR,
        estimate_params,
    )

    st.markdown(f'<div class="section-title">🌱 {t("crop_title",L)}</div>', unsafe_allow_html=True)
    st.caption("Answer a few simple questions — no technical knowledge needed.")

    # ── Shared styles ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .step-indicator { display:flex; gap:8px; margin: 1rem 0 0.4rem 0; }
    .step-dot { width:32px; height:32px; border-radius:50%; display:flex; align-items:center;
                justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0; }
    .step-dot.done   { background:#2d6a4f; color:white; }
    .step-dot.active { background:#52b788; color:white; box-shadow:0 0 0 3px #bbf7d0; }
    .step-dot.todo   { background:#e8f5e9; color:#aaa; border:2px solid #c8e6c9; }
    .step-line { flex:1; height:2px; background:#e8f5e9; margin:auto; }
    .step-line.done  { background:#2d6a4f; }
    .step-labels { display:flex; font-size:0.7rem; color:#777; margin-bottom:1.5rem; }
    .step-label  { flex:1; text-align:center; }
    .q-card { background:white; border:1.5px solid #bbf7d0; border-radius:14px;
               padding:1.5rem 1.8rem; margin:1rem 0; box-shadow:0 2px 10px rgba(26,71,49,0.06); }
    .q-title { color:#1a4731; font-size:1.05rem; font-weight:600; margin-bottom:0.3rem; }
    .q-sub   { color:#555; font-size:0.92rem; font-weight:400; margin-bottom:1rem; }
    .env-pill { display:inline-block; background:#f0fdf4; border:1px solid #86efac;
                border-radius:8px; padding:0.3rem 0.8rem; margin:0.2rem; font-size:0.85rem; color:#166534; }
    .crop-rank-card { background:white; border-radius:12px; padding:1.2rem 1.5rem;
                      margin:0.6rem 0; border-left:5px solid #52b788;
                      box-shadow:0 1px 6px rgba(26,71,49,0.08); }
    .crop-rank-card.secondary { border-left-color:#e9c46a; opacity:0.92; }
    .crop-rank-card.tertiary  { border-left-color:#adb5bd; opacity:0.85; }
    .crop-rank-num  { font-size:0.75rem; font-weight:700; color:#888; text-transform:uppercase;
                      letter-spacing:0.05em; margin-bottom:4px; }
    .crop-rank-name { font-size:1.3rem; font-weight:700; color:#1a4731;
                      font-family:'Playfair Display',serif; text-transform:capitalize; }
    .conf-bar-wrap  { background:#e8f5e9; border-radius:99px; height:8px; margin:8px 0 4px 0; overflow:hidden; }
    .conf-bar       { height:8px; border-radius:99px; background:linear-gradient(90deg,#2d6a4f,#52b788); }
    .conf-bar.yellow{ background:linear-gradient(90deg,#d4a017,#e9c46a); }
    .conf-bar.grey  { background:linear-gradient(90deg,#6c757d,#adb5bd); }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state initialisation ──────────────────────────────────────────
    for key, default in [
        ("crop_step",        0),
        ("sel_state",        None),
        ("sel_district",     None),
        ("soil_detected",    None),
        ("_last_img",        None),
        ("ans_N",            None),
        ("ans_P1",           None),
        ("ans_P2",           None),
        ("ans_K1",           None),
        ("ans_K2",           None),
        ("ans_rain",         None),
        ("crop_result",      None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    step = st.session_state.crop_step

    # ── Step indicator helper (4 visible steps) ───────────────────────────────
    # Internal steps → display bucket:
    #   0          → bucket 0  (Location)
    #   1          → bucket 1  (Soil)
    #   2,3,4,5    → bucket 2  (Questions)
    #   5.5        → bucket 2  (computing)
    #   6          → bucket 3  (Results)
    if step == 0:
        ds = 0
    elif step == 1:
        ds = 1
    elif step == 6:
        ds = 3
    else:
        ds = 2

    def _dot(n):
        if n < ds:  return f'<div class="step-dot done">✓</div>'
        if n == ds: return f'<div class="step-dot active">{n+1}</div>'
        return          f'<div class="step-dot todo">{n+1}</div>'

    def _line(n):
        cls = "done" if n < ds else ""
        return f'<div class="step-line {cls}"></div>'

    st.markdown(f"""
    <div class="step-indicator">
        {_dot(0)}{_line(0)}{_dot(1)}{_line(1)}{_dot(2)}{_line(2)}{_dot(3)}
    </div>
    <div class="step-labels">
        <span class="step-label">📍 Location</span>
        <span class="step-label">🪨 Soil</span>
        <span class="step-label">📋 Questions</span>
        <span class="step-label">✅ Results</span>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0 — State & District Selection
    # ══════════════════════════════════════════════════════════════════════════
    if step == 0:
        st.markdown("### 📍 Where is your farm located?")
        st.caption("We'll use your district to fetch local weather and calibrate rainfall estimates.")

        states_dict = get_states_and_districts()
        states      = sorted(states_dict.keys())

        col1, col2 = st.columns(2)
        with col1:
            sel_state = st.selectbox("🗺️ State", ["— Select State —"] + states, key="state_sel")
        with col2:
            if sel_state != "— Select State —":
                districts    = states_dict.get(sel_state, [])
                sel_district = st.selectbox("📍 District", ["— Select District —"] + districts, key="dist_sel")
            else:
                st.selectbox("📍 District", ["— Select a state first —"],
                             disabled=True, key="dist_sel_disabled")
                sel_district = "— Select District —"

        if st.button("Continue →", key="step0_next"):
            if sel_state == "— Select State —":
                st.warning("Please select your state.")
            elif sel_district == "— Select District —":
                st.warning("Please select your district.")
            else:
                st.session_state.sel_state    = sel_state
                st.session_state.sel_district = sel_district
                st.session_state.crop_step    = 1
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Soil Image Upload
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 1:
        from utils.soil_utils import detect_soil_type, SOIL_TYPES

        district = st.session_state.sel_district
        state    = st.session_state.sel_state

        # Location badge
        st.info(f"📍 **{district}, {state}**")

        # Fetch environmental data silently (used later for prediction)
        env         = get_district_env_data(district)
        annual_rain = get_district_annual_rainfall(district)

        st.markdown("#### 📷 Upload a photo of your soil")
        st.caption("Dry or slightly moist soil works best. Make sure the soil is clearly visible.")

        soil_image = st.file_uploader("", type=["jpg", "jpeg", "png"], key="soil_img")

        if not soil_image:
            st.info("👆 Upload a clear photo of your soil to continue.")
        else:
            img_bytes = soil_image.read()

            # Detect once per image
            if (st.session_state.soil_detected is None
                    or st.session_state.get("_last_img") != soil_image.name):
                with st.spinner("🔍 Analyzing soil type..."):
                    detected_soil, soil_conf, avg_rgb, method = detect_soil_type(img_bytes)
                if soil_conf < 50.0:
                    st.error("❌ Couldn't identify a soil type from this image. "
                             "Please try a clearer photo of the soil surface.")
                    st.stop()
                st.session_state.soil_detected = (detected_soil, soil_conf)
                st.session_state["_last_img"]  = soil_image.name
                # Reset any downstream answers
                for k in ["ans_N", "ans_P1", "ans_P2", "ans_K1", "ans_K2", "ans_rain", "crop_result"]:
                    st.session_state[k] = None

            detected_soil, soil_conf = st.session_state.soil_detected
            info = SOIL_TYPES.get(detected_soil, {})

            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(soil_image, caption="Your Soil Sample", use_container_width=True)
            with col_info:
                st.markdown(f"""
                <div class="result-box" style="margin-top:0">
                    <h2 style="font-size:1.3rem">🪨 {detected_soil}</h2>
                    <div class="crop-detail">
                        {info.get('description','')}<br><br>
                        <strong>Characteristics:</strong> {info.get('characteristics','')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_back, col_next = st.columns([1, 3])
            with col_back:
                if st.button("← Back", key="back_step1"):
                    st.session_state.crop_step = 0
                    st.rerun()
            with col_next:
                if st.button("Continue with this soil →", key="step1_next"):
                    st.session_state.crop_step = 2
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Nitrogen: Leaf Colour
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 2:
        detected_soil, soil_conf = st.session_state.soil_detected
        st.caption(f"🪨 {detected_soil} · 📍 {st.session_state.sel_district}")
        st.markdown("---")

        st.markdown(f"""
        <div class="q-card">
        <div class="q-title">🌿 Question 1 of 6 &nbsp;·&nbsp; Nitrogen Estimation</div>
        <div class="q-sub">How do the crop leaves usually look in your field?</div>
        </div>
        """, unsafe_allow_html=True)

        opts = [
            ("Yellowish",     "🟡", "Pale / yellow leaves — low nitrogen"),
            ("Light green",   "🍃", "Light but not yellow — mild deficiency"),
            ("Healthy green", "🌿", "Normal healthy green — adequate nitrogen"),
            ("Dark green",    "🌲", "Very dark green — nitrogen is plenty"),
        ]
        cols = st.columns(4)
        for i, (val, icon, hint) in enumerate(opts):
            with cols[i]:
                if st.button(f"{icon} {val}", key=f"n_{i}", use_container_width=True, help=hint):
                    st.session_state.ans_N     = val
                    st.session_state.crop_step = 3
                    st.rerun()

        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Back", key="back_step2"):
                st.session_state.crop_step = 1
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Phosphorus Q1: Early Growth
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3:
        st.caption(f"✅ N answer: **{st.session_state.ans_N}**")
        st.markdown("---")

        st.markdown("""
        <div class="q-card">
        <div class="q-title">🌱 Question 2 of 6 &nbsp;·&nbsp; Phosphorus Estimation</div>
        <div class="q-sub">Did the crop grow well in the early stage (germination / seedling)?</div>
        </div>
        """, unsafe_allow_html=True)
        p1_opts = ["Weak growth", "Okay growth", "Good growth", "Very healthy growth"]
        p1_sel  = st.radio("", p1_opts, key="p1_radio", horizontal=True, label_visibility="collapsed")

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if st.button("← Back", key="back_step3"):
                st.session_state.crop_step = 2
                st.rerun()
        with col_next:
            if st.button("Next →", key="step3_next"):
                st.session_state.ans_P1    = p1_sel
                st.session_state.crop_step = 3.5
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3.5 — Phosphorus Q2: Flowering Delay
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3.5:
        st.caption(f"✅ N: **{st.session_state.ans_N}** · P Growth: **{st.session_state.ans_P1}**")
        st.markdown("---")

        st.markdown("""
        <div class="q-card">
        <div class="q-title">🌸 Question 3 of 6 &nbsp;·&nbsp; Phosphorus Estimation</div>
        <div class="q-sub">Does the crop take too long to grow or flower (delayed maturity)?</div>
        </div>
        """, unsafe_allow_html=True)
        p2_opts = ["Yes", "Sometimes", "No"]
        p2_sel  = st.radio("", p2_opts, key="p2_radio", horizontal=True, label_visibility="collapsed")

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if st.button("← Back", key="back_step3b"):
                st.session_state.crop_step = 3
                st.rerun()
        with col_next:
            if st.button("Next →", key="step3b_next"):
                st.session_state.ans_P2    = p2_sel
                st.session_state.crop_step = 4
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Potassium Q1: Drought Response
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 4:
        st.caption(f"✅ N: **{st.session_state.ans_N}** · P Growth: **{st.session_state.ans_P1}**")
        st.markdown("---")

        st.markdown("""
        <div class="q-card">
        <div class="q-title">💧 Question 4 of 6 &nbsp;·&nbsp; Potassium Estimation</div>
        <div class="q-sub">What happens to the crop when water is limited for a few days?</div>
        </div>
        """, unsafe_allow_html=True)
        k1_opts = ["Plants dry quickly", "Plants become weak", "Plants manage somehow", "Plants stay healthy"]
        k1_sel  = st.radio("", k1_opts, key="k1_radio", horizontal=False, label_visibility="collapsed")

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if st.button("← Back", key="back_step4"):
                st.session_state.crop_step = 3.5
                st.rerun()
        with col_next:
            if st.button("Next →", key="step4_next"):
                st.session_state.ans_K1    = k1_sel
                st.session_state.crop_step = 4.5
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4.5 — Potassium Q2: Tip Burn
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 4.5:
        st.caption(f"✅ N: **{st.session_state.ans_N}** · K Drought: **{st.session_state.ans_K1}**")
        st.markdown("---")

        st.markdown("""
        <div class="q-card">
        <div class="q-title">🍂 Question 5 of 6 &nbsp;·&nbsp; Potassium Estimation</div>
        <div class="q-sub">Do the leaves dry out or turn brown starting from the edges / sides?</div>
        </div>
        """, unsafe_allow_html=True)
        k2_opts = ["Yes, very often", "Sometimes", "Rarely", "Never noticed"]
        k2_sel  = st.radio("", k2_opts, key="k2_radio", horizontal=True, label_visibility="collapsed")

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if st.button("← Back", key="back_step4b"):
                st.session_state.crop_step = 4
                st.rerun()
        with col_next:
            if st.button("Next →", key="step4b_next"):
                st.session_state.ans_K2    = k2_sel
                st.session_state.crop_step = 5
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Rainfall Feedback (district-calibrated)
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 5:
        district    = st.session_state.sel_district
        annual_avg  = get_district_annual_rainfall(district)
        monthly_avg = annual_avg / 12.0

        st.caption(f"✅ N: **{st.session_state.ans_N}** · "
                   f"P: **{st.session_state.ans_P1}** · "
                   f"K: **{st.session_state.ans_K1}**")
        st.markdown("---")

        # Build district-calibrated labels
        rain_labels = get_rainfall_option_labels(district)

        st.markdown(f"""
        <div class="q-card">
        <div class="q-title">🌧️ Question 6 of 6 &nbsp;·&nbsp; Rainfall Feedback</div>
        <div class="q-sub">How has the rain been in your area recently?<br>
        <em style="color:#888;font-size:0.83rem">
        Your district's long-term average: ~{monthly_avg:.0f} mm/month ({annual_avg:.0f} mm/year)</em></div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        for i, opt in enumerate(["Very low", "Less than usual", "Normal", "Heavy"]):
            label = rain_labels[opt]
            with cols[i]:
                if st.button(label, key=f"rain_{i}", use_container_width=True):
                    st.session_state.ans_rain  = opt
                    st.session_state.crop_step = 5.5
                    st.rerun()

        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Back", key="back_step5"):
                st.session_state.crop_step = 4.5
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5.5 — Auto-compute (triggered immediately after rainfall answer)
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.crop_step == 5.5:
        detected_soil, _ = st.session_state.soil_detected
        with st.spinner("🌾 Estimating soil parameters and predicting best crops…"):
            try:
                params = estimate_params(
                    soil_type = detected_soil,
                    district  = st.session_state.sel_district,
                    ans_N     = st.session_state.ans_N,
                    ans_P1    = st.session_state.ans_P1,
                    ans_P2    = st.session_state.ans_P2,
                    ans_K1    = st.session_state.ans_K1,
                    ans_K2    = st.session_state.ans_K2,
                    ans_rain  = st.session_state.ans_rain,
                )
                top_crops = predict_top_crops_merged(
                    params["N"], params["P"], params["K"],
                    params["temperature"], params["humidity"],
                    params["ph"], params["rainfall"],
                    soil_type = detected_soil,
                    top_n=3, threshold=5.0,
                )
                st.session_state.crop_result = (top_crops, params)
                st.session_state.crop_step   = 6
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.crop_step = 5

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Results
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.crop_step == 6 and st.session_state.crop_result:
        top_crops, params     = st.session_state.crop_result
        detected_soil, s_conf = st.session_state.soil_detected
        district              = st.session_state.sel_district

        st.markdown("---")
        st.markdown(f"### 🌾 Recommended Crop for {district}")
        st.caption("Based on your soil, district climate, and field observations.")

        # Show only the top crop (highest confidence)
        best = top_crops[0]
        if isinstance(best, dict):
            crop = best["crop"]
        else:
            crop = best[0]

        st.markdown(f"""
        <div class="crop-rank-card">
            <div class="crop-rank-num">🥇 Best Match</div>
            <div class="crop-rank-name">{crop.title()}</div>
        </div>
        """, unsafe_allow_html=True)

        # Parameter details expander
        with st.expander("🔍 View estimated parameters used for prediction"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N (kg/ha)",  params["N"])
            c2.metric("P (kg/ha)",  params["P"])
            c3.metric("K (kg/ha)",  params["K"])
            c4.metric("pH",         params["ph"])
            c1b, c2b, c3b = st.columns(3)
            c1b.metric("Temperature (°C)", params["temperature"])
            c2b.metric("Humidity (%)",      params["humidity"])
            c3b.metric("Rainfall (mm/mo)", params["rainfall"])
            st.caption(
                f"🪨 Soil: {detected_soil} &nbsp;·&nbsp; "
                f"📍 District: {district} &nbsp;·&nbsp; "
                f"🌡️ Environmental data from district records"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Start Over", key="restart"):
            for key in ["crop_step", "sel_state", "sel_district", "soil_detected", "_last_img",
                        "ans_N", "ans_P1", "ans_P2", "ans_K1", "ans_K2", "ans_rain", "crop_result"]:
                st.session_state.pop(key, None)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# FERTILIZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("nav_fertilizer", L):
    st.markdown(f'<div class="section-title">🧪 {t("fertilizer_title",L)}</div>', unsafe_allow_html=True)

    crops = get_all_crops()

    st.markdown('<div class="fert-input-card">', unsafe_allow_html=True)
    st.markdown("#### 🌾 Select your crop and soil type")
    st.caption("That's all we need — we'll handle the rest.")

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌱 Crop", crops, help="Choose the crop you want to grow")
    with col2:
        soil_options = get_soil_types_for_crop(crop)
        soil_type = st.selectbox("🪨 Soil Type", soil_options, help="Select your field's soil type")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Get Fertilizer Recommendation", use_container_width=True):
        result = recommend_fertilizer(crop, soil_type)

        if result:
            st.markdown(f"""
            <div class="fert-result-header">
                <h2>🌿 {result['crop']} on {result['soil_type']} Soil</h2>
                <p>📅 Season: {result['season']} &nbsp;|&nbsp; 🪨 Soil: {result['soil_type']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="fert-dose-box">
                <h4>💊 Recommended Fertilizer</h4>
                <p>{result['fertilizer_name']}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="fert-dose-box" style="border-color:#86efac">
                    <h4 style="color:#166534">📦 Dose</h4>
                    <p>{result['fertilizer_dose']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="fert-dose-box" style="border-color:#93c5fd">
                    <h4 style="color:#1e40af">🔧 Application Method</h4>
                    <p>{result['application_method']}</p>
                </div>
                """, unsafe_allow_html=True)

            npk = result['ideal_npk']
            st.markdown("**📊 Ideal Soil Parameters for this combination:**")
            st.markdown(f"""
            <span class="npk-pill npk-n">N: {npk['N']} kg/ha</span>
            <span class="npk-pill npk-p">P: {npk['P']} kg/ha</span>
            <span class="npk-pill npk-k">K: {npk['K']} kg/ha</span>
            <span class="npk-pill npk-ph">pH: {npk['pH']}</span>
            <span class="npk-pill npk-w">Moisture: {npk['soil_moisture']}%</span>
            """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ No fertilizer data found for this crop and soil combination. Try a different soil type.")

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("nav_disease", L):
    st.markdown(f'<div class="section-title">🔬 {t("disease_title",L)}</div>', unsafe_allow_html=True)

    SUPPORTED_CROPS = [
        "Apple", "Blueberry", "Cherry", "Corn (Maize)", "Grape",
        "Orange", "Peach", "Pepper (Bell)", "Potato", "Raspberry",
        "Soybean", "Squash", "Strawberry", "Tomato",
    ]

    with st.expander("ℹ️ Supported crops"):
        st.markdown(", ".join(SUPPORTED_CROPS))

    uploaded = st.file_uploader(t("upload_image", L), type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="Uploaded Leaf Image", use_container_width=True)

        with col2:
            if st.button(t("detect_disease", L)):
                with st.spinner(t("loading", L)):
                    try:
                        from utils.disease_utils import predict_disease, SEVERITY_COLOR, NotALeafError, ModelNotFoundError
                        img_bytes = uploaded.read()
                        class_key, display_name, confidence, info = predict_disease(img_bytes)

                        if confidence < 75:
                            st.warning(
                                "⚠️ Could not identify the disease with enough confidence. "
                                "Please upload a clearer, well-lit photo of the affected leaf."
                            )
                        else:
                            severity   = info.get('severity', 'Unknown')
                            sev_color  = SEVERITY_COLOR.get(severity, '#555')
                            is_healthy = severity == 'None'
                            box_icon   = "✅" if is_healthy else "⚠️"

                            st.markdown(f"""
                            <div class="result-box">
                                <h2>{box_icon} {display_name}</h2>
                                <div class="crop-detail">
                                    <span style="background:{sev_color};color:white;padding:2px 10px;border-radius:12px;font-size:0.82rem;font-weight:600">{severity} Severity</span>
                                    <br><br>
                                    🦠 <strong>Cause:</strong> {info.get('cause','N/A')}<br><br>
                                    🔍 <strong>Symptoms:</strong> {info.get('symptoms','N/A')}<br><br>
                                    🛡️ <strong>Prevention/Treatment:</strong> {info.get('prevention','N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    except ImportError:
                        st.warning("⚙️ PyTorch is not installed. Run: `pip install torch torchvision` to enable this module.")
                    except NotALeafError:
                        st.warning("🌿 The uploaded image does not appear to be a plant leaf. Please upload a clear, close-up photo of an affected leaf.")
                    except ModelNotFoundError:
                        st.error("⚙️ Disease detection model not found. Please train the model first using `train_disease_model.py`.")
                    except ValueError as e:
                        st.warning(f"🖼️ {str(e)}")
                    except Exception as e:
                        st.error("❌ Something went wrong during detection. Please try again with a different image.")

# ══════════════════════════════════════════════════════════════════════════════

