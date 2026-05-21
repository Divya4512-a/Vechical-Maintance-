import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Vehicle Monitoring System",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================
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


maintenance_model, cost_model, fuel_model = load_models()

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#071739,#0b2b5b);
    color:white;
}

h1,h2,h3{
    color:#00d4ff;
}

[data-testid="stSidebar"]{
    background:#04152d;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#00d4ff,#0066ff);
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#0066ff,#00d4ff);
}

.metric-box{
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.2);
}

.big-font{
    font-size:35px;
    font-weight:bold;
}

.small-font{
    color:#cccccc;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>
🚗 AI Vehicle Monitoring System
</h1>
<p style='text-align:center;color:#cccccc;'>
Smart Maintenance Prediction Dashboard
</p>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.title("⚙ Vehicle Details")

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    ["Sedan", "SUV", "Bike", "Scooter", "Truck"]
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Electric", "Hybrid"]
)

route_type = st.sidebar.selectbox(
    "Route Type",
    ["Urban", "Rural", "Highway"]
)

vehicle_age = st.sidebar.slider(
    "Vehicle Age",
    0, 30, 5
)

mileage = st.sidebar.slider(
    "Mileage",
    0, 300000, 20000
)

usage_hours = st.sidebar.slider(
    "Usage Hours",
    0, 5000, 500
)

driving_score = st.sidebar.slider(
    "Driving Score",
    0, 100, 75
)

load_capacity = st.sidebar.slider(
    "Load Capacity",
    0, 5000, 1000
)

engine_temp = st.sidebar.slider(
    "Engine Temperature",
    40, 150, 85
)

tire_pressure = st.sidebar.slider(
    "Tire Pressure",
    20, 60, 32
)

battery_health = st.sidebar.slider(
    "Battery Health",
    0, 100, 80
)

fuel_before = st.sidebar.number_input(
    "Fuel Consumption Before",
    1.0, 50.0, 12.0
)

predict_btn = st.sidebar.button("⚡ Analyze Vehicle")

# =========================================================
# ENCODING
# =========================================================
vehicle_map = {
    "Bike":0,
    "Scooter":1,
    "Sedan":2,
    "SUV":3,
    "Truck":4
}

fuel_map = {
    "Petrol":0,
    "Diesel":1,
    "Electric":2,
    "Hybrid":3
}

route_map = {
    "Urban":0,
    "Rural":1,
    "Highway":2
}

# =========================================================
# PREDICTION
# =========================================================
if predict_btn:

    input_data = pd.DataFrame([{
        "Vehicle_Type": vehicle_map[vehicle_type],
        "Fuel_Type": fuel_map[fuel_type],
        "Route_Type": route_map[route_type],
        "Vehicle_Age": vehicle_age,
        "Mileage": mileage,
        "Usage_Hours": usage_hours,
        "Driving_Score": driving_score,
        "Load_Capacity": load_capacity,
        "Engine_Temperature": engine_temp,
        "Tire_Pressure": tire_pressure,
        "Battery_Health": battery_health,
        "Fuel_Consumption_Before": fuel_before
    }])

    prediction = maintenance_model.predict(input_data)[0]

    if prediction == 1:
        estimated_cost = cost_model.predict(input_data)[0]
        fuel_after = fuel_model.predict(input_data)[0]
        status = "Maintenance Required"
        emoji = "⚠"
    else:
        estimated_cost = 0
        fuel_after = fuel_before
        status = "Vehicle Healthy"
        emoji = "✅"

    # =====================================================
    # RESULT CARDS
    # =====================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='big-font'>{emoji}</div>
            <h3>Status</h3>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='big-font'>₹ {estimated_cost:.0f}</div>
            <h3>Estimated Cost</h3>
            <p class='small-font'>Repair Cost</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='big-font'>{fuel_after:.2f}</div>
            <h3>Fuel After Service</h3>
            <p class='small-font'>L/100km</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # SUMMARY TABLE
    # =====================================================
    summary = pd.DataFrame({
        "Parameter":[
            "Vehicle Type",
            "Fuel Type",
            "Route Type",
            "Vehicle Age",
            "Mileage",
            "Usage Hours",
            "Driving Score",
            "Load Capacity",
            "Engine Temperature",
            "Tire Pressure",
            "Battery Health",
            "Fuel Before"
        ],
        "Value":[
            vehicle_type,
            fuel_type,
            route_type,
            vehicle_age,
            mileage,
            usage_hours,
            driving_score,
            load_capacity,
            engine_temp,
            tire_pressure,
            battery_health,
            fuel_before
        ]
    })

    st.subheader("📋 Vehicle Summary")
    st.dataframe(summary, width='stretch', hide_index=True)

    # =====================================================
    # RECOMMENDATION
    # =====================================================
    st.subheader("🛠 Recommendations")

    if prediction == 1:

        st.warning("""
        • Check engine oil
        
        • Inspect tire pressure
        
        • Check battery
        
        • Service engine
        
        • Visit nearest service center
        """)

    else:

        st.success("""
        • Vehicle is in good condition
        
        • Continue regular maintenance
        
        • Monitor fuel usage
        
        • Recheck after few months
        """)

else:

    st.info("⬅ Fill vehicle details from sidebar and click ANALYZE VEHICLE")