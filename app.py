"""
╔══════════════════════════════════════════════════════════════╗
║         AI Vehicle Monitoring System — Streamlit App         ║
║         Premium Futuristic Dashboard UI                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Vehicle Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    maintenance_model = joblib.load(
        r"C:\Users\Divya\OneDrive\Desktop\ai_ml\project\datasets\vehicle_maintenance_model.pkl"
    )
    cost_model = joblib.load(
        r"C:\Users\Divya\OneDrive\Desktop\ai_ml\project\datasets\maintenance_cost_model.pkl"
    )
    fuel_model = joblib.load(
        r"C:\Users\Divya\OneDrive\Desktop\ai_ml\project\datasets\fuel_consumption_after_model.pkl"
    )
    return maintenance_model, cost_model, fuel_model

maintenance_model, cost_model, fuel_model = load_models()

# ──────────────────────────────────────────────
# SESSION STATE — for reset after prediction
# ──────────────────────────────────────────────
defaults = {
    "vehicle_type": "Sedan",
    "fuel_type": "Petrol",
    "route_type": "Urban",
    "vehicle_age": 3,
    "mileage": 15000,
    "usage_hours": 500,
    "driving_score": 75,
    "load_capacity": 500,
    "engine_temp": 85,
    "tire_pressure": 32,
    "battery_health": 80,
    "fuel_before": 12.0,
    "show_results": False,
    "prediction": None,
    "est_cost": 0.0,
    "fuel_after": 0.0,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# GLOBAL CSS — Dark futuristic glassmorphism theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Exo+2:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #050a14;
    --bg-secondary:  #0a1628;
    --accent-cyan:   #00d4ff;
    --accent-blue:   #0066ff;
    --accent-green:  #00ff9d;
    --accent-amber:  #ffaa00;
    --accent-red:    #ff4466;
    --glass-bg:      rgba(255,255,255,0.04);
    --glass-border:  rgba(0,212,255,0.18);
    --text-primary:  #e8f4ff;
    --text-muted:    rgba(180,210,255,0.6);
    --glow-cyan:     0 0 20px rgba(0,212,255,0.35);
    --glow-green:    0 0 20px rgba(0,255,157,0.35);
    --glow-red:      0 0 20px rgba(255,68,102,0.35);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* ── App Background ── */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%,   rgba(0,102,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,212,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 50%,  rgba(0,255,157,0.04) 0%, transparent 70%),
        linear-gradient(180deg, #050a14 0%, #060d1e 50%, #040810 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,22,40,0.98) 0%, rgba(5,10,20,0.98) 100%) !important;
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Sidebar Labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--accent-cyan) !important;
}

/* ── Selectbox ── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(0,212,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Sliders ── */
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
    font-weight: 700 !important;
    font-family: 'Exo 2', sans-serif !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: var(--accent-cyan) !important;
    border: 2px solid white !important;
    box-shadow: var(--glow-cyan) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ── Slider track ── */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div[role="progressbar"],
[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* ── Number Input ── */
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(0,212,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    caret-color: transparent !important;
    user-select: none !important;
    cursor: default !important;
    text-align: center !important;
}
[data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stNumberInput button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    color: var(--accent-cyan) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover {
    background: rgba(0,212,255,0.25) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Main Content Headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.5rem !important;
    color: var(--accent-cyan) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid var(--glass-border) !important;
}
.dataframe thead tr th {
    background: rgba(0,102,255,0.25) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
}
.dataframe tbody tr td {
    background: rgba(5,10,20,0.7) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(0,212,255,0.08) !important;
}
.dataframe tbody tr:hover td {
    background: rgba(0,212,255,0.06) !important;
}

/* ── Success / Warning / Info ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ── Analyze button ── */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    padding: 0.85rem 1rem !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--bg-primary) !important;
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.45), 0 4px 20px rgba(0,102,255,0.4) !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(0,212,255,0.65), 0 8px 30px rgba(0,102,255,0.5) !important;
}

/* ── Main area buttons (reset) ── */
.stButton > button {
    font-family: 'Exo 2', sans-serif !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    background: rgba(0,212,255,0.08) !important;
    color: var(--accent-cyan) !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.18) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ===== GLOBAL CURSOR REMOVAL FIX ===== */

/* Remove blinking text cursor everywhere */
html, body, [class*="css"],
input, textarea, select,
div, span, p, h1, h2, h3, h4, h5, h6,
button,
label,
[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"],
[data-testid="stAppViewContainer"],
.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stSlider div {
    caret-color: transparent !important;
    user-select: none !important;
    cursor: default !important;
}

/* Remove focus cursor */
*:focus,
*:active,
input:focus,
textarea:focus,
select:focus,
button:focus {
    caret-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Streamlit-specific fixes */
input,
textarea {
    caret-color: transparent !important;
}

[data-baseweb="select"] * {
    caret-color: transparent !important;
}

[data-testid="stAppViewContainer"] * {
    caret-color: transparent !important;
}

/* Remove I-beam text cursor on title/text */
h1, h2, h3, p, span, div {
    cursor: default !important;
}

/* Keep buttons clickable */
button {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPER — glass card HTML
# ──────────────────────────────────────────────
def glass_card(icon, title, value, sub, color="#00d4ff", glow_color="rgba(0,212,255,0.35)"):
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid {color}44;
        border-radius: 20px;
        padding: 1.6rem 1.4rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 30px {glow_color}, inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
        height: 100%;
    ">
        <div style="
            position: absolute; top:-20px; right:-20px;
            width:100px; height:100px;
            background: radial-gradient(circle, {color}22, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="font-size:2.2rem; margin-bottom:0.6rem;">{icon}</div>
        <div style="
            font-family:'Exo 2',sans-serif;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.1em;
            text-transform:uppercase;
            color:{color};
            margin-bottom:0.5rem;
            opacity:0.85;
        ">{title}</div>
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:1.55rem;
            font-weight:800;
            color:#e8f4ff;
            line-height:1.2;
            margin-bottom:0.35rem;
        ">{value}</div>
        <div style="
            font-family:'Exo 2',sans-serif;
            font-size:0.8rem;
            color:rgba(180,210,255,0.55);
        ">{sub}</div>
    </div>
    """

# ──────────────────────────────────────────────
# SIDEBAR — Input Panel
# ──────────────────────────────────────────────
with st.sidebar:
    # Sidebar header
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem; text-align:center;">
        <div style="font-size:2.4rem; margin-bottom:0.4rem;">🚗</div>
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.18em;
            text-transform:uppercase;
            color:#00d4ff;
            line-height:1.6;
        ">Vehicle Input<br>
        <span style="color:rgba(0,212,255,0.45);font-size:0.62rem;">CONTROL PANEL</span>
        </div>
        <div style="
            height:2px;
            background:linear-gradient(90deg,transparent,#00d4ff,transparent);
            margin-top:0.9rem;
        "></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section: Vehicle Identity ──
    st.markdown('<p style="font-family:Orbitron,sans-serif;font-size:0.65rem;letter-spacing:0.15em;color:rgba(0,212,255,0.5);text-transform:uppercase;margin-bottom:0.5rem;">⬡ Vehicle Identity</p>', unsafe_allow_html=True)

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Sedan", "SUV", "Bike", "Scooter", "Auto Rickshaw", "Truck", "Van"],
        index=["Sedan", "SUV", "Bike", "Scooter", "Auto Rickshaw", "Truck", "Van"].index(st.session_state.vehicle_type),
        key="sb_vehicle_type"
    )
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"],
        index=["Petrol", "Diesel", "Electric", "Hybrid", "CNG"].index(st.session_state.fuel_type),
        key="sb_fuel_type"
    )
    route_type = st.selectbox(
        "Route Type",
        ["Urban", "Rural", "Highway"],
        index=["Urban", "Rural", "Highway"].index(st.session_state.route_type),
        key="sb_route_type"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section: Vehicle Stats ──
    st.markdown('<p style="font-family:Orbitron,sans-serif;font-size:0.65rem;letter-spacing:0.15em;color:rgba(0,212,255,0.5);text-transform:uppercase;margin-bottom:0.5rem;">⬡ Vehicle Stats</p>', unsafe_allow_html=True)

    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, st.session_state.vehicle_age, key="sl_vehicle_age")
    mileage     = st.slider("Mileage (km)", 0, 300000, st.session_state.mileage, step=500, key="sl_mileage")
    usage_hours = st.slider("Usage Hours", 0, 5000, st.session_state.usage_hours, step=10, key="sl_usage_hours")
    driving_score = st.slider("Driving Score", 0, 100, st.session_state.driving_score, key="sl_driving_score")
    load_capacity = st.slider("Load Capacity (kg)", 0, 5000, st.session_state.load_capacity, step=10, key="sl_load_capacity")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section: Diagnostics ──
    st.markdown('<p style="font-family:Orbitron,sans-serif;font-size:0.65rem;letter-spacing:0.15em;color:rgba(0,212,255,0.5);text-transform:uppercase;margin-bottom:0.5rem;">⬡ Diagnostics</p>', unsafe_allow_html=True)

    engine_temp   = st.slider("Engine Temperature (°C)", 40, 150, st.session_state.engine_temp, key="sl_engine_temp")
    tire_pressure = st.slider("Tire Pressure (PSI)", 20, 60, st.session_state.tire_pressure, key="sl_tire_pressure")
    battery_health = st.slider("Battery Health (%)", 0, 100, st.session_state.battery_health, key="sl_battery_health")
    fuel_before   = st.number_input(
        "Fuel Consumption Before (L/100km)",
        min_value=1.0, max_value=50.0,
        value=float(st.session_state.fuel_before),
        step=0.5, format="%.1f",
        key="ni_fuel_before"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyze Button ──
    analyze_clicked = st.button("⚡ ANALYZE VEHICLE", key="btn_analyze")

# ──────────────────────────────────────────────
# ENCODINGS
# ──────────────────────────────────────────────
vehicle_map = {"Bike":0, "Scooter":1, "Auto Rickshaw":2, "Sedan":3, "SUV":4, "Truck":5, "Van":6}
fuel_map    = {"Petrol":0, "Diesel":1, "Electric":2, "Hybrid":3, "CNG":4}
route_map   = {"Urban":0, "Rural":1, "Highway":2}

# ──────────────────────────────────────────────
# PREDICTION LOGIC
# ──────────────────────────────────────────────
if analyze_clicked:
    # Build feature DataFrame in exact model order
    input_data = pd.DataFrame([{
        "Vehicle_Type":         vehicle_map[vehicle_type],
        "Fuel_Type":            fuel_map[fuel_type],
        "Route_Type":           route_map[route_type],
        "Vehicle_Age":          vehicle_age,
        "Mileage":              mileage,
        "Usage_Hours":          usage_hours,
        "Driving_Score":        driving_score,
        "Load_Capacity":        load_capacity,
        "Engine_Temperature":   engine_temp,
        "Tire_Pressure":        tire_pressure,
        "Battery_Health":       battery_health,
        "Fuel_Consumption_Before": fuel_before,
    }])

    prediction = maintenance_model.predict(input_data)[0]

    if prediction == 1:
        est_cost   = float(cost_model.predict(input_data)[0])
        fuel_after = float(fuel_model.predict(input_data)[0])
    else:
        est_cost   = 0.0
        fuel_after = fuel_before

    # Store results
    st.session_state.prediction = int(prediction)
    st.session_state.est_cost   = est_cost
    st.session_state.fuel_after = fuel_after
    st.session_state.show_results = True

    # Save inputs for summary table
    st.session_state.last_inputs = {
        "vehicle_type":   vehicle_type,
        "fuel_type":      fuel_type,
        "route_type":     route_type,
        "vehicle_age":    vehicle_age,
        "mileage":        mileage,
        "usage_hours":    usage_hours,
        "driving_score":  driving_score,
        "load_capacity":  load_capacity,
        "engine_temp":    engine_temp,
        "tire_pressure":  tire_pressure,
        "battery_health": battery_health,
        "fuel_before":    fuel_before,
    }

    # Reset sidebar state to defaults
    for k, v in defaults.items():
        if k not in ("show_results", "prediction", "est_cost", "fuel_after"):
            st.session_state[k] = v

    st.rerun()

# ──────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────

# ── Hero Title ──
st.markdown("""
<div style="text-align:center; padding: 2rem 0 0.5rem;">
    <div style="
        font-family:'Orbitron',sans-serif;
        font-size:clamp(1.5rem,3.5vw,2.6rem);
        font-weight:900;
        letter-spacing:0.04em;
        background: linear-gradient(135deg, #0066ff 0%, #00d4ff 50%, #00ff9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height:1.2;
        margin-bottom:0.4rem;
    ">
        🚗 AI VEHICLE MONITORING SYSTEM
    </div>
    <div style="
        font-family:'Exo 2',sans-serif;
        font-size:0.95rem;
        font-weight:400;
        color:rgba(180,210,255,0.55);
        letter-spacing:0.22em;
        text-transform:uppercase;
    ">
        Smart Vehicle Prediction Dashboard
    </div>
    <div style="
        width:180px;
        height:3px;
        background:linear-gradient(90deg,#0066ff,#00d4ff,#00ff9d);
        border-radius:2px;
        margin:0.9rem auto 0;
        box-shadow:0 0 18px rgba(0,212,255,0.6);
    "></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# RESULTS SECTION
# ──────────────────────────────────────────────
if st.session_state.show_results and st.session_state.prediction is not None:

    pred      = st.session_state.prediction
    est_cost  = st.session_state.est_cost
    fuel_after= st.session_state.fuel_after
    inputs    = st.session_state.get("last_inputs", {})

    # ── Status Banner ──
    if pred == 1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255,68,102,0.15), rgba(255,170,0,0.1));
            border: 1px solid rgba(255,68,102,0.4);
            border-left: 4px solid #ff4466;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            display:flex; align-items:center; gap:1rem;
            margin-bottom:1.5rem;
            box-shadow: 0 0 30px rgba(255,68,102,0.2);
        ">
            <span style="font-size:2rem;">⚠️</span>
            <div>
                <div style="font-family:'Orbitron',sans-serif;font-size:0.85rem;font-weight:700;color:#ff4466;letter-spacing:0.1em;">MAINTENANCE ALERT DETECTED</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.8rem;color:rgba(255,180,180,0.7);margin-top:0.2rem;">
                    Your vehicle requires immediate attention. Review the diagnostics below.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0,255,157,0.12), rgba(0,212,255,0.07));
            border: 1px solid rgba(0,255,157,0.35);
            border-left: 4px solid #00ff9d;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            display:flex; align-items:center; gap:1rem;
            margin-bottom:1.5rem;
            box-shadow: 0 0 30px rgba(0,255,157,0.15);
        ">
            <span style="font-size:2rem;">✅</span>
            <div>
                <div style="font-family:'Orbitron',sans-serif;font-size:0.85rem;font-weight:700;color:#00ff9d;letter-spacing:0.1em;">SYSTEM NOMINAL — ALL CLEAR</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.8rem;color:rgba(180,255,220,0.7);margin-top:0.2rem;">
                    No maintenance required at this time. Vehicle is performing optimally.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Result Metric Cards ──
    c1, c2, c3 = st.columns(3)

    with c1:
        status_icon  = "🔴" if pred == 1 else "🟢"
        status_txt   = "REQUIRED" if pred == 1 else "NOT REQUIRED"
        status_sub   = "Immediate service advised" if pred == 1 else "Vehicle running smoothly"
        status_color = "#ff4466" if pred == 1 else "#00ff9d"
        st.markdown(glass_card(status_icon, "Maintenance Status", status_txt, status_sub, color=status_color,
                               glow_color="rgba(255,68,102,0.3)" if pred == 1 else "rgba(0,255,157,0.3)"),
                    unsafe_allow_html=True)

    with c2:
        cost_txt = f"₹ {est_cost:,.0f}" if pred == 1 else "₹ 0"
        cost_sub = "Estimated service cost" if pred == 1 else "No expense needed"
        st.markdown(glass_card("💰", "Estimated Cost", cost_txt, cost_sub,
                               color="#ffaa00", glow_color="rgba(255,170,0,0.3)"),
                    unsafe_allow_html=True)

    with c3:
        fuel_txt = f"{fuel_after:.2f} L/100km"
        fuel_sub = "After servicing" if pred == 1 else "Current consumption (unchanged)"
        st.markdown(glass_card("⛽", "Fuel After Service", fuel_txt, fuel_sub,
                               color="#00d4ff", glow_color="rgba(0,212,255,0.3)"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vehicle Summary Table ──
    st.markdown("""
    <div style="
        font-family:'Orbitron',sans-serif;
        font-size:0.8rem;
        font-weight:700;
        letter-spacing:0.14em;
        text-transform:uppercase;
        color:#00d4ff;
        margin-bottom:0.8rem;
    ">📋 Vehicle Input Summary</div>
    """, unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Parameter": [
            "🚗 Vehicle Type", "⛽ Fuel Type", "🛣️ Route Type",
            "📅 Vehicle Age", "🛞 Mileage", "⏱️ Usage Hours",
            "🎯 Driving Score", "⚖️ Load Capacity",
            "🌡️ Engine Temp", "🔵 Tire Pressure",
            "🔋 Battery Health", "💧 Fuel Consumption Before"
        ],
        "Value": [
            inputs.get("vehicle_type","—"),
            inputs.get("fuel_type","—"),
            inputs.get("route_type","—"),
            f"{inputs.get('vehicle_age','—')} yrs",
            f"{inputs.get('mileage','—'):,} km",
            f"{inputs.get('usage_hours','—')} hrs",
            f"{inputs.get('driving_score','—')} / 100",
            f"{inputs.get('load_capacity','—')} kg",
            f"{inputs.get('engine_temp','—')} °C",
            f"{inputs.get('tire_pressure','—')} PSI",
            f"{inputs.get('battery_health','—')} %",
            f"{inputs.get('fuel_before','—')} L/100km",
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
            "Value":     st.column_config.TextColumn("Value",     width="medium"),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendation Box ──
    if pred == 1:
        rec_html = """
        <div style="
            background:rgba(255,170,0,0.07);
            border:1px solid rgba(255,170,0,0.3);
            border-radius:14px;
            padding:1.2rem 1.5rem;
            box-shadow:0 0 20px rgba(255,170,0,0.12);
        ">
            <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;font-weight:700;color:#ffaa00;letter-spacing:0.12em;margin-bottom:0.7rem;">
                🔧 MAINTENANCE RECOMMENDATIONS
            </div>
            <ul style="font-family:'Exo 2',sans-serif;font-size:0.85rem;color:rgba(255,220,150,0.85);line-height:2;margin:0;padding-left:1.2rem;">
                <li>Schedule a full vehicle inspection at an authorized service center</li>
                <li>Check engine oil levels and replace if overdue</li>
                <li>Inspect and calibrate tire pressure to manufacturer specs</li>
                <li>Test battery health and charging system</li>
                <li>Review brake pads, filters, and fluid levels</li>
                <li>Verify engine temperature sensors are functioning correctly</li>
            </ul>
        </div>
        """
    else:
        rec_html = """
        <div style="
            background:rgba(0,255,157,0.05);
            border:1px solid rgba(0,255,157,0.25);
            border-radius:14px;
            padding:1.2rem 1.5rem;
            box-shadow:0 0 20px rgba(0,255,157,0.08);
        ">
            <div style="font-family:'Orbitron',sans-serif;font-size:0.75rem;font-weight:700;color:#00ff9d;letter-spacing:0.12em;margin-bottom:0.7rem;">
                ✅ PREVENTIVE TIPS — KEEP IT RUNNING GREAT
            </div>
            <ul style="font-family:'Exo 2',sans-serif;font-size:0.85rem;color:rgba(150,255,210,0.8);line-height:2;margin:0;padding-left:1.2rem;">
                <li>Continue following the scheduled maintenance calendar</li>
                <li>Monitor fuel consumption trends regularly</li>
                <li>Keep tire pressure within the recommended range</li>
                <li>Maintain smooth driving habits to preserve driving score</li>
                <li>Re-analyze after every 5,000 km or 3 months for best results</li>
            </ul>
        </div>
        """

    st.markdown(rec_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reset Button ──
    col_r = st.columns([3, 1, 3])
    with col_r[1]:
        if st.button("🔄 New Analysis", key="btn_reset"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

# ──────────────────────────────────────────────
# IDLE / WELCOME STATE
# ──────────────────────────────────────────────
else:
    # ── Welcome Cards ──
    st.markdown("""
    <div style="text-align:center;margin:1rem 0 2rem;">
        <div style="
            font-family:'Exo 2',sans-serif;
            font-size:0.95rem;
            color:rgba(180,210,255,0.45);
            letter-spacing:0.08em;
        ">
            Configure vehicle parameters in the sidebar and click <strong style="color:#00d4ff;">ANALYZE VEHICLE</strong> to generate predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(glass_card("🔍", "Maintenance Detection",
                               "ML Powered", "Predicts service needs instantly",
                               "#00d4ff"), unsafe_allow_html=True)
    with col2:
        st.markdown(glass_card("💰", "Cost Estimation",
                               "AI Accurate", "Estimates repair expenses",
                               "#ffaa00", "rgba(255,170,0,0.3)"), unsafe_allow_html=True)
    with col3:
        st.markdown(glass_card("⛽", "Fuel Prediction",
                               "Post-Service", "Forecasts efficiency after repair",
                               "#00ff9d", "rgba(0,255,157,0.3)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(0,212,255,0.12);
        border-radius:18px;
        padding:1.8rem 2rem;
        backdrop-filter:blur(10px);
    ">
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:0.8rem;
            font-weight:700;
            letter-spacing:0.14em;
            color:#00d4ff;
            margin-bottom:1.2rem;
            text-transform:uppercase;
        ">⚙️ How It Works</div>
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:180px;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">1️⃣</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.85rem;font-weight:600;color:#e8f4ff;margin-bottom:0.3rem;">Input Vehicle Data</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.78rem;color:rgba(180,210,255,0.5);">Fill in all 12 vehicle parameters using the sidebar controls</div>
            </div>
            <div style="flex:1;min-width:180px;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">2️⃣</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.85rem;font-weight:600;color:#e8f4ff;margin-bottom:0.3rem;">AI Analysis</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.78rem;color:rgba(180,210,255,0.5);">Three ML models evaluate your vehicle's health in real-time</div>
            </div>
            <div style="flex:1;min-width:180px;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">3️⃣</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.85rem;font-weight:600;color:#e8f4ff;margin-bottom:0.3rem;">Smart Insights</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.78rem;color:rgba(180,210,255,0.5);">Get maintenance status, cost estimate, and fuel efficiency forecast</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    text-align:center;
    padding:1rem 0;
    border-top:1px solid rgba(0,212,255,0.1);
">
    <span style="
        font-family:'Exo 2',sans-serif;
        font-size:0.72rem;
        letter-spacing:0.12em;
        text-transform:uppercase;
        color:rgba(180,210,255,0.25);
    ">
        AI Vehicle Monitoring System &nbsp;|&nbsp; Powered by Machine Learning &nbsp;|&nbsp; 🚗 Smart Prediction Dashboard
    </span>
</div>
""", unsafe_allow_html=True)