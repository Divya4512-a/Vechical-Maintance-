import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS | AI Vehicle Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
@st.cache_resource
def load_models():

    maintenance_model = joblib.load(
        "datasets/vehicle_maintenance_model.pkl"
    )

    cost_model = joblib.load(
        "datasets/maintenance_cost_model.pkl"
    )

    fuel_model = joblib.load(
        "datasets/fuel_consumption_after_model.pkl"
    )

    return maintenance_model, cost_model, fuel_model
# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Root variables ── */
:root {
  --neon-blue:   #00d4ff;
  --neon-cyan:   #00ffea;
  --neon-purple: #7b2fff;
  --neon-green:  #00ff88;
  --neon-red:    #ff3366;
  --neon-orange: #ff8c00;
  --bg-void:     #020408;
  --bg-deep:     #040d1a;
  --bg-card:     rgba(4, 20, 40, 0.85);
  --bg-glass:    rgba(0, 212, 255, 0.04);
  --border-glow: rgba(0, 212, 255, 0.25);
  --text-primary:   #e8f4fd;
  --text-secondary: #7ab8d4;
  --text-muted:     #3a6b85;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif;
  background-color: var(--bg-void) !important;
  color: var(--text-primary) !important;
}

/* Animated grid background */
.stApp {
  background:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px),
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,255,0.08) 0%, transparent 60%),
    #020408 !important;
  background-size: 60px 60px, 60px 60px, 100% 100% !important;
  animation: gridScroll 20s linear infinite;
}

@keyframes gridScroll {
  0%   { background-position: 0 0, 0 0, 50% -20%; }
  100% { background-position: 0 60px, 60px 0, 50% -20%; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--neon-blue); border-radius: 2px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020c1a 0%, #010810 100%) !important;
  border-right: 1px solid var(--border-glow) !important;
  box-shadow: 4px 0 30px rgba(0,212,255,0.08) !important;
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-blue), var(--neon-cyan), transparent);
  animation: scanLine 3s ease-in-out infinite;
}
@keyframes scanLine {
  0%, 100% { opacity: 0.4; }
  50%       { opacity: 1; }
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {
  color: var(--neon-cyan) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
}

/* Sliders */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--neon-blue) !important;
  box-shadow: 0 0 12px var(--neon-blue) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
  background: rgba(0,212,255,0.2) !important;
}

/* Select boxes */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
  background: rgba(0,212,255,0.06) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div:focus-within {
  border-color: var(--neon-blue) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* Number inputs */
[data-testid="stSidebar"] input[type="number"] {
  background: rgba(0,212,255,0.06) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  font-family: 'Share Tech Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,255,234,0.08)) !important;
  border: 1px solid var(--neon-blue) !important;
  color: var(--neon-cyan) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 0.6rem 1.4rem !important;
  transition: all 0.3s ease !important;
  position: relative !important;
  overflow: hidden !important;
  width: 100% !important;
}
.stButton > button::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent);
  transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(0,255,234,0.15)) !important;
  box-shadow: 0 0 25px rgba(0,212,255,0.5), 0 0 50px rgba(0,212,255,0.2) !important;
  transform: translateY(-1px) !important;
  color: #fff !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Dividers ── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent) !important;
  margin: 1.5rem 0 !important;
}

/* ── Tab overrides (used for section nav) ── */
[data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 4px !important;
}
[data-baseweb="tab"] {
  background: rgba(0,212,255,0.04) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 4px !important;
  color: var(--text-secondary) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.1em !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: rgba(0,212,255,0.12) !important;
  border-color: var(--neon-blue) !important;
  color: var(--neon-cyan) !important;
  box-shadow: 0 0 12px rgba(0,212,255,0.3) !important;
}
[data-baseweb="tab-highlight"] { background: transparent !important; }

/* ── Metric boxes ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
  backdrop-filter: blur(12px) !important;
}
[data-testid="stMetric"] label {
  color: var(--text-secondary) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.6rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  color: var(--neon-cyan) !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 1.6rem !important;
}
[data-testid="stMetricDelta"] { font-family: 'Rajdhani', sans-serif !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-glow) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── Animated top banner ───────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(90deg, #020c1a, #041830, #020c1a);
  border-bottom: 1px solid rgba(0,212,255,0.2);
  padding: 0;
  margin: 0 -2rem 1.5rem -2rem;
  overflow: hidden;
  position: relative;
  height: 3px;
">
  <div style="
    position: absolute; top: 0; left: -50%;
    width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, #00d4ff, #00ffea, transparent);
    animation: bannerScan 2.5s linear infinite;
  "></div>
</div>
<style>
@keyframes bannerScan {
  0%   { left: -50%; }
  100% { left: 150%; }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.2rem 0 0.8rem;
  border-bottom: 1px solid rgba(0,212,255,0.12);
  margin-bottom: 1.8rem;
">
  <div style="display: flex; align-items: center; gap: 1.2rem;">
    <div style="
      width: 52px; height: 52px;
      background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,255,234,0.1));
      border: 1px solid rgba(0,212,255,0.4);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.6rem;
      box-shadow: 0 0 20px rgba(0,212,255,0.3), inset 0 0 20px rgba(0,212,255,0.05);
    ">🚗</div>
    <div>
      <div style="
        font-family: 'Orbitron', monospace;
        font-size: 1.55rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        background: linear-gradient(90deg, #00d4ff, #00ffea, #00d4ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
      ">NEXUS</div>
      <div style="
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.72rem;
        color: #3a6b85;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        margin-top: -4px;
      ">AI Vehicle Intelligence Platform</div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 1.5rem;">
    <div style="text-align: right;">
      <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: #3a6b85; letter-spacing: 0.2em;">SYSTEM STATUS</div>
      <div style="display: flex; align-items: center; gap: 6px; justify-content: flex-end; margin-top: 2px;">
        <div style="
          width: 8px; height: 8px; border-radius: 50%;
          background: #00ff88;
          box-shadow: 0 0 8px #00ff88;
          animation: pulse 1.5s ease-in-out infinite;
        "></div>
        <span style="font-family: 'Orbitron', monospace; font-size: 0.62rem; color: #00ff88; letter-spacing: 0.15em;">ONLINE</span>
      </div>
    </div>
    <div style="text-align: right;">
      <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: #3a6b85; letter-spacing: 0.2em;">MODELS LOADED</div>
      <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; color: #00d4ff; letter-spacing: 0.1em; margin-top: 2px;">3 / 3 ACTIVE</div>
    </div>
  </div>
</div>
<style>
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
@keyframes pulse {
  0%,100%{ box-shadow: 0 0 4px #00ff88; }
  50%    { box-shadow: 0 0 14px #00ff88, 0 0 28px rgba(0,255,136,0.4); }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
      <div style="
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.25em;
        color: #00d4ff;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(0,212,255,0.15);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
      ">⚙ Vehicle Parameters</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:Orbitron,monospace;font-size:0.6rem;letter-spacing:0.2em;color:#3a6b85;text-transform:uppercase;margin-bottom:0.3rem;">Vehicle Profile</p>', unsafe_allow_html=True)

    vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
    mileage     = st.slider("Mileage (km)", 0, 300000, 80000, step=1000)
    engine_size = st.selectbox("Engine Size (L)", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.2),transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Orbitron,monospace;font-size:0.6rem;letter-spacing:0.2em;color:#3a6b85;text-transform:uppercase;margin-bottom:0.3rem;">Performance Data</p>', unsafe_allow_html=True)

    fuel_efficiency     = st.slider("Fuel Efficiency (km/L)", 5.0, 25.0, 15.0, step=0.5)
    last_service_days   = st.slider("Days Since Last Service", 0, 730, 180)
    brake_condition_pct = st.slider("Brake Pad Condition (%)", 0, 100, 60)
    tire_tread_mm       = st.slider("Tire Tread Depth (mm)", 0.0, 10.0, 5.0, step=0.1)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.2),transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Orbitron,monospace;font-size:0.6rem;letter-spacing:0.2em;color:#3a6b85;text-transform:uppercase;margin-bottom:0.3rem;">Operational Data</p>', unsafe_allow_html=True)

    accident_history    = st.selectbox("Accident History", ["None", "Minor", "Moderate", "Major"])
    fuel_type           = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
    transmission        = st.selectbox("Transmission", ["Manual", "Automatic"])
    avg_speed           = st.slider("Average Speed (km/h)", 20, 120, 60)
    ac_usage_pct        = st.slider("A/C Usage (%)", 0, 100, 50)
    load_weight_kg      = st.slider("Typical Load (kg)", 0, 500, 150)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.2),transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)

    run_analysis = st.button("⚡  RUN AI ANALYSIS")

# ── Encode categoricals ───────────────────────────────────────────────────────
accident_map     = {"None": 0, "Minor": 1, "Moderate": 2, "Major": 3}
fuel_type_map    = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3}
transmission_map = {"Manual": 0, "Automatic": 1}

accident_enc     = accident_map[accident_history]
fuel_type_enc    = fuel_type_map[fuel_type]
transmission_enc = transmission_map[transmission]

features = np.array([[
    vehicle_age, mileage, engine_size, fuel_efficiency,
    last_service_days, brake_condition_pct, tire_tread_mm,
    accident_enc, fuel_type_enc, transmission_enc,
    avg_speed, ac_usage_pct, load_weight_kg
]])

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Vehicle Age", f"{vehicle_age} yrs", f"{'⚠ Aging' if vehicle_age > 10 else '✓ Prime'}")
with c2:
    st.metric("Mileage", f"{mileage:,} km", f"{'⚠ High' if mileage > 150000 else '✓ Normal'}")
with c3:
    st.metric("Fuel Efficiency", f"{fuel_efficiency} km/L", f"{'↑ Good' if fuel_efficiency > 15 else '↓ Low'}")
with c4:
    st.metric("Brake Condition", f"{brake_condition_pct}%", f"{'⚠ Replace' if brake_condition_pct < 30 else '✓ OK'}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Helper: styled section header ─────────────────────────────────────────────
def section_header(icon, title, subtitle):
    st.markdown(f"""
    <div style="
      display: flex; align-items: center; gap: 14px;
      margin: 2rem 0 1.2rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid rgba(0,212,255,0.12);
    ">
      <div style="
        font-size: 1.4rem;
        width: 42px; height: 42px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 8px;
      ">{icon}</div>
      <div>
        <div style="font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;
          letter-spacing:0.1em;color:#e8f4fd;">{title}</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:0.75rem;
          color:#3a6b85;letter-spacing:0.15em;text-transform:uppercase;">{subtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction result card ─────────────────────────────────────────────────────
def result_card(title, value, unit, status, status_color, icon, detail=""):
    color_map = {
        "green":  ("#00ff88", "rgba(0,255,136,0.08)", "rgba(0,255,136,0.3)"),
        "yellow": ("#ffd700", "rgba(255,215,0,0.08)", "rgba(255,215,0,0.3)"),
        "red":    ("#ff3366", "rgba(255,51,102,0.08)", "rgba(255,51,102,0.3)"),
        "blue":   ("#00d4ff", "rgba(0,212,255,0.08)", "rgba(0,212,255,0.3)"),
    }
    fg, bg, glow = color_map.get(status_color, color_map["blue"])
    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, var(--bg-card), rgba(0,212,255,0.03));
      border: 1px solid {glow};
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 4px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
      position: relative;
      overflow: hidden;
      transition: transform 0.3s ease;
    ">
      <!-- corner accent -->
      <div style="
        position: absolute; top: 0; right: 0;
        width: 60px; height: 60px;
        background: radial-gradient(circle at top right, {glow}, transparent 70%);
      "></div>
      <!-- bottom bar -->
      <div style="
        position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, {fg}, transparent);
      "></div>

      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;">
        <div>
          <div style="font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:0.2em;
            color:#3a6b85;text-transform:uppercase;margin-bottom:0.5rem;">{title}</div>
          <div style="
            font-family:'Share Tech Mono',monospace;
            font-size:2rem; font-weight:700;
            color:{fg};
            text-shadow: 0 0 20px {fg}, 0 0 40px rgba({','.join(str(int(fg.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.4);
            line-height:1;
          ">{value}<span style="font-size:0.9rem;margin-left:4px;color:#7ab8d4;">{unit}</span></div>
          {'<div style="font-family:Rajdhani,sans-serif;font-size:0.78rem;color:#7ab8d4;margin-top:0.4rem;">' + detail + '</div>' if detail else ''}
        </div>
        <div style="
          width:46px;height:46px;border-radius:10px;
          background:{bg};border:1px solid {glow};
          display:flex;align-items:center;justify-content:center;
          font-size:1.4rem;flex-shrink:0;
        ">{icon}</div>
      </div>

      <div style="
        display:inline-flex;align-items:center;gap:6px;
        background:{bg};border:1px solid {glow};
        border-radius:4px;padding:3px 10px;margin-top:1rem;
      ">
        <div style="width:6px;height:6px;border-radius:50%;background:{fg};
          box-shadow:0 0 6px {fg};"></div>
        <span style="font-family:'Orbitron',monospace;font-size:0.58rem;
          letter-spacing:0.15em;color:{fg};">{status}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main logic: idle vs results ───────────────────────────────────────────────
if not run_analysis:
    # ── Idle dashboard ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      text-align: center;
      padding: 3.5rem 2rem;
      background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(0,255,234,0.02));
      border: 1px dashed rgba(0,212,255,0.18);
      border-radius: 16px;
      position: relative;
      overflow: hidden;
    ">
      <div style="
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 300px; height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
        animation: breathe 4s ease-in-out infinite;
      "></div>
      <div style="font-size: 3.5rem; margin-bottom: 1rem; position: relative;">🔬</div>
      <div style="
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #e8f4fd;
        margin-bottom: 0.5rem;
        position: relative;
      ">READY FOR ANALYSIS</div>
      <div style="
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        color: #3a6b85;
        letter-spacing: 0.1em;
        position: relative;
      ">Configure vehicle parameters in the sidebar and run AI analysis to get predictive insights</div>
    </div>
    <style>
    @keyframes breathe {
      0%,100% { transform: translate(-50%,-50%) scale(1); opacity:0.5; }
      50%      { transform: translate(-50%,-50%) scale(1.3); opacity:1; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Feature overview cards ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.65rem;letter-spacing:0.25em;
      color:#3a6b85;text-transform:uppercase;text-align:center;margin-bottom:1.2rem;">
      AI Models Available
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    for col, icon, title, desc, clr in [
        (fc1, "🔧", "MAINTENANCE PREDICTOR", "Predicts if maintenance is needed based on vehicle condition", "#00ff88"),
        (fc2, "💰", "COST ESTIMATOR", "Forecasts upcoming maintenance and repair costs", "#ffd700"),
        (fc3, "⛽", "FUEL ANALYZER", "Predicts future fuel consumption patterns", "#00d4ff"),
    ]:
        with col:
            col.markdown(f"""
            <div style="
              background: rgba(4,20,40,0.7);
              border: 1px solid rgba(0,212,255,0.12);
              border-top: 2px solid {clr};
              border-radius: 10px;
              padding: 1.3rem;
              text-align: center;
            ">
              <div style="font-size:2rem;margin-bottom:0.75rem;">{icon}</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.62rem;
                letter-spacing:0.15em;color:{clr};margin-bottom:0.5rem;">{title}</div>
              <div style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;color:#3a6b85;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Loading animation ─────────────────────────────────────────────────────
    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        steps = [
            (20, "Initializing neural network..."),
            (40, "Processing vehicle telemetry..."),
            (60, "Running maintenance model..."),
            (80, "Calculating cost projections..."),
            (100, "Generating AI insights..."),
        ]
        for pct, msg in steps:
            status_text.markdown(f"""
            <div style="
              font-family:'Share Tech Mono',monospace;
              font-size:0.75rem;color:#00d4ff;
              letter-spacing:0.15em;text-align:center;
              padding:0.3rem;
            ">▶ {msg}</div>
            """, unsafe_allow_html=True)
            progress_bar.progress(pct)
            time.sleep(0.18)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()

    # ── Run predictions ───────────────────────────────────────────────────────
    maintenance_pred = maintenance_model.predict(features)[0]
    cost_pred        = cost_model.predict(features)[0]
    fuel_pred        = fuel_model.predict(features)[0]

    # ── Section 1: Core predictions ───────────────────────────────────────────
    section_header("🧠", "PREDICTIVE ANALYSIS", "AI Model Output — Real-time Inference")

    r1, r2, r3 = st.columns(3)
    with r1:
        if maintenance_pred == 1:
            result_card("Maintenance Required", "YES", "", "ACTION NEEDED", "red", "🔧", "Immediate service recommended")
        else:
            result_card("Maintenance Required", "NO", "", "CONDITION NORMAL", "green", "✅", "Vehicle is in good condition")
    with r2:
        cost_label  = "BUDGET ALERT" if cost_pred > 15000 else ("MODERATE COST" if cost_pred > 5000 else "LOW COST")
        cost_color  = "red" if cost_pred > 15000 else ("yellow" if cost_pred > 5000 else "green")
        result_card("Est. Maintenance Cost", f"₹{cost_pred:,.0f}", "", cost_label, cost_color, "💰",
                    f"{'High spend — plan ahead' if cost_pred > 15000 else 'Manageable cost range'}")
    with r3:
        fuel_label = "HIGH CONSUMPTION" if fuel_pred > 10 else ("AVERAGE" if fuel_pred > 7 else "EFFICIENT")
        fuel_color = "red" if fuel_pred > 10 else ("yellow" if fuel_pred > 7 else "green")
        result_card("Predicted Fuel Usage", f"{fuel_pred:.2f}", "L/100km", fuel_label, fuel_color, "⛽",
                    f"Est. ₹{fuel_pred * 95:.0f}/100km at current prices")

    # ── Section 2: Vehicle health radar ──────────────────────────────────────
    section_header("📊", "VEHICLE HEALTH MATRIX", "Multi-dimensional condition assessment")

    def health_bar(label, value, max_val, good_threshold, warn_threshold, unit="", invert=False):
        pct = min(value / max_val * 100, 100)
        if invert:
            color = "#ff3366" if value < warn_threshold else ("#ffd700" if value < good_threshold else "#00ff88")
        else:
            color = "#ff3366" if value < warn_threshold else ("#ffd700" if value < good_threshold else "#00ff88")
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:'Rajdhani',sans-serif;font-size:0.78rem;
              color:#7ab8d4;letter-spacing:0.08em;text-transform:uppercase;">{label}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:{color};">
              {value}{unit}</span>
          </div>
          <div style="background:rgba(255,255,255,0.05);border-radius:2px;height:6px;overflow:hidden;">
            <div style="
              width:{pct}%;height:100%;border-radius:2px;
              background:linear-gradient(90deg,{color}aa,{color});
              box-shadow:0 0 8px {color};
              transition:width 1s ease;
            "></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown("""
        <div style="
          background:rgba(4,20,40,0.7);border:1px solid rgba(0,212,255,0.12);
          border-radius:10px;padding:1.3rem;
        ">
        <div style="font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:0.2em;
          color:#3a6b85;text-transform:uppercase;margin-bottom:1rem;">Component Health</div>
        """, unsafe_allow_html=True)
        health_bar("Brake Pads",    brake_condition_pct, 100, 60, 30, "%")
        health_bar("Tire Tread",    tire_tread_mm, 10, 4, 2, "mm")
        health_bar("Fuel Economy",  fuel_efficiency, 25, 12, 8, " km/L")
        st.markdown("</div>", unsafe_allow_html=True)

    with hc2:
        st.markdown("""
        <div style="
          background:rgba(4,20,40,0.7);border:1px solid rgba(0,212,255,0.12);
          border-radius:10px;padding:1.3rem;
        ">
        <div style="font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:0.2em;
          color:#3a6b85;text-transform:uppercase;margin-bottom:1rem;">Usage Metrics</div>
        """, unsafe_allow_html=True)
        health_bar("Service Interval", max(0, 730 - last_service_days), 730, 365, 180, " days left")
        health_bar("Age Factor",       max(0, 20 - vehicle_age), 20, 10, 5, " yrs left")
        health_bar("Mileage Buffer",   max(0, 300000 - mileage), 300000, 150000, 80000, " km left")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Summary table ───────────────────────────────────────────────
    section_header("📋", "DIAGNOSTIC SUMMARY", "Complete vehicle telemetry report")

    df = pd.DataFrame({
        "Parameter":       ["Vehicle Age", "Mileage", "Engine Size", "Fuel Efficiency",
                             "Days Since Service", "Brake Condition", "Tire Tread",
                             "Accident History", "Fuel Type", "Transmission",
                             "Avg Speed", "A/C Usage", "Load Weight"],
        "Value":           [f"{vehicle_age} yrs", f"{mileage:,} km", f"{engine_size} L",
                             f"{fuel_efficiency} km/L", f"{last_service_days} days",
                             f"{brake_condition_pct}%", f"{tire_tread_mm} mm",
                             accident_history, fuel_type, transmission,
                             f"{avg_speed} km/h", f"{ac_usage_pct}%", f"{load_weight_kg} kg"],
        "Status":          [
            "⚠ Aging" if vehicle_age > 10 else "✓ Normal",
            "⚠ High"  if mileage > 150000 else "✓ Normal",
            "✓ Standard", "✓ Good" if fuel_efficiency > 15 else "⚠ Low",
            "⚠ Due"   if last_service_days > 365 else "✓ OK",
            "⚠ Low"   if brake_condition_pct < 30 else "✓ OK",
            "⚠ Worn"  if tire_tread_mm < 2 else "✓ OK",
            "⚠ Risk"  if accident_enc >= 2 else "✓ Clear",
            "✓ Logged", "✓ Logged",
            "⚠ Fast"  if avg_speed > 100 else "✓ Normal",
            "⚠ High"  if ac_usage_pct > 80 else "✓ Normal",
            "⚠ Heavy" if load_weight_kg > 400 else "✓ Normal",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Section 4: AI Recommendations ─────────────────────────────────────────
    section_header("💡", "AI RECOMMENDATIONS", "Intelligent action plan based on diagnostics")

    recommendations = []
    if maintenance_pred == 1:
        recommendations.append(("🔧", "URGENT: Schedule maintenance immediately", "red",
                                  "AI has detected that your vehicle requires immediate attention. Delaying service may lead to component failure."))
    if cost_pred > 15000:
        recommendations.append(("💰", "HIGH COST ALERT: Budget ₹{:,.0f}".format(cost_pred), "yellow",
                                  "Predicted maintenance cost is significant. Consider phased service or compare quotes from multiple service centers."))
    if brake_condition_pct < 30:
        recommendations.append(("🛑", "CRITICAL: Replace brake pads urgently", "red",
                                  "Brake pad condition is below safe threshold. This is a safety-critical issue requiring immediate replacement."))
    if tire_tread_mm < 2:
        recommendations.append(("🔄", "Replace tires immediately", "red",
                                  "Tire tread depth is dangerously low. Risk of hydroplaning and loss of traction in wet conditions."))
    if last_service_days > 365:
        recommendations.append(("📅", "Overdue for service", "yellow",
                                  "Your vehicle hasn't been serviced in over a year. Schedule a comprehensive inspection."))
    if fuel_pred > 10:
        recommendations.append(("⛽", "High fuel consumption detected", "yellow",
                                  "Consider tuning the engine, checking air filter, and verifying tire pressure to improve efficiency."))
    if not recommendations:
        recommendations.append(("✅", "Vehicle is in excellent condition", "green",
                                  "All systems nominal. Continue regular maintenance schedule and monitoring."))

    color_map = {"red": "#ff3366", "yellow": "#ffd700", "green": "#00ff88", "blue": "#00d4ff"}
    for icon, title, clr, desc in recommendations:
        fg = color_map[clr]
        st.markdown(f"""
        <div style="
          background: linear-gradient(135deg, rgba(4,20,40,0.9), rgba(0,212,255,0.02));
          border: 1px solid rgba({','.join(str(int(fg.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.3);
          border-left: 3px solid {fg};
          border-radius: 8px;
          padding: 1rem 1.3rem;
          margin-bottom: 0.75rem;
          display: flex;
          align-items: flex-start;
          gap: 1rem;
        ">
          <div style="font-size:1.3rem;flex-shrink:0;margin-top:1px;">{icon}</div>
          <div>
            <div style="font-family:'Orbitron',monospace;font-size:0.68rem;font-weight:700;
              letter-spacing:0.1em;color:{fg};margin-bottom:0.3rem;">{title}</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:0.85rem;color:#7ab8d4;
              line-height:1.5;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Success banner ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      margin-top: 2rem;
      background: linear-gradient(135deg, rgba(0,255,136,0.06), rgba(0,212,255,0.03));
      border: 1px solid rgba(0,255,136,0.25);
      border-radius: 10px;
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    ">
      <div style="
        width:10px;height:10px;border-radius:50%;background:#00ff88;
        box-shadow:0 0 12px #00ff88;flex-shrink:0;
        animation: pulse 1.5s ease-in-out infinite;
      "></div>
      <div>
        <span style="font-family:'Orbitron',monospace;font-size:0.65rem;
          letter-spacing:0.15em;color:#00ff88;">ANALYSIS COMPLETE</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;
          color:#3a6b85;margin-left:1rem;">
          All 3 AI models executed successfully · Results based on real-time inference
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
  border-top: 1px solid rgba(0,212,255,0.1);
  padding: 1.5rem 0 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
">
  <div>
    <span style="font-family:'Orbitron',monospace;font-size:0.75rem;
      font-weight:700;color:#00d4ff;letter-spacing:0.15em;">NEXUS</span>
    <span style="font-family:'Rajdhani',sans-serif;font-size:0.75rem;
      color:#3a6b85;margin-left:0.75rem;letter-spacing:0.1em;">
      AI Vehicle Intelligence Platform · v2.0
    </span>
  </div>
  <div style="display:flex;gap:2rem;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a6b85;">
      Models: Maintenance · Cost · Fuel
    </span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a6b85;">
      Engine: Scikit-Learn · Streamlit
    </span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a6b85;">
      Powered by Machine Learning
    </span>
  </div>
</div>
""", unsafe_allow_html=True)