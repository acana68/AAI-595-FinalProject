"""
ERTime - ER Wait Time Predictor
AAI 595 - Applied Machine Learning
Michael Lugo, Amoy Mowatt, Ardit Cana, Wesley Nabo

Run with: streamlit run app.py
"""

import math
import warnings
from datetime import datetime

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERTime - ER Wait Time Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Sample hospital database ───────────────────────────────────────────────────
HOSPITALS = pd.DataFrame([
    # Urban – East Coast
    {"name": "Metro General Hospital",      "city": "New York, NY",      "lat": 40.7391, "lon": -73.9755, "region": "Urban", "facility_size": 180, "nurse_ratio": 2, "specialist_avail": 9},
    {"name": "Riverside Medical Center",    "city": "New York, NY",      "lat": 40.7644, "lon": -73.9566, "region": "Urban", "facility_size": 200, "nurse_ratio": 2, "specialist_avail": 10},
    {"name": "Harbor View Hospital",        "city": "Brooklyn, NY",      "lat": 40.6501, "lon": -73.9496, "region": "Urban", "facility_size": 160, "nurse_ratio": 3, "specialist_avail": 7},
    {"name": "Capital Health ER",           "city": "Washington, DC",    "lat": 38.9072, "lon": -77.0369, "region": "Urban", "facility_size": 175, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Liberty Regional Hospital",   "city": "Philadelphia, PA",  "lat": 39.9526, "lon": -75.1652, "region": "Urban", "facility_size": 165, "nurse_ratio": 3, "specialist_avail": 7},
    # Urban – Midwest
    {"name": "Lakeside Medical Center",     "city": "Chicago, IL",       "lat": 41.8955, "lon": -87.6218, "region": "Urban", "facility_size": 190, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Great Lakes ER",              "city": "Detroit, MI",       "lat": 42.3314, "lon": -83.0458, "region": "Urban", "facility_size": 155, "nurse_ratio": 3, "specialist_avail": 6},
    # Urban – South
    {"name": "Southern Regional Hospital",  "city": "Atlanta, GA",       "lat": 33.7490, "lon": -84.3880, "region": "Urban", "facility_size": 165, "nurse_ratio": 3, "specialist_avail": 7},
    {"name": "Gulf Coast Medical Center",   "city": "Houston, TX",       "lat": 29.7604, "lon": -95.3698, "region": "Urban", "facility_size": 185, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Bayshore University Hospital","city": "Miami, FL",         "lat": 25.7617, "lon": -80.1918, "region": "Urban", "facility_size": 170, "nurse_ratio": 2, "specialist_avail": 9},
    # Urban – West Coast
    {"name": "Pacific Coast Hospital",      "city": "Los Angeles, CA",   "lat": 34.0522, "lon": -118.2437,"region": "Urban", "facility_size": 195, "nurse_ratio": 2, "specialist_avail": 9},
    {"name": "Bay Area Medical Center",     "city": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194,"region": "Urban", "facility_size": 175, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Cascades Regional Hospital",  "city": "Seattle, WA",       "lat": 47.6062, "lon": -122.3321,"region": "Urban", "facility_size": 160, "nurse_ratio": 3, "specialist_avail": 7},
    # Hudson County NJ (Hoboken / Jersey City area)
    {"name": "Hoboken University Medical Center", "city": "Hoboken, NJ",    "lat": 40.7440, "lon": -74.0324, "region": "Urban", "facility_size": 130, "nurse_ratio": 3, "specialist_avail": 6},
    {"name": "Jersey City Medical Center",        "city": "Jersey City, NJ", "lat": 40.7178, "lon": -74.0431, "region": "Urban", "facility_size": 175, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Christ Hospital Jersey City",       "city": "Jersey City, NJ", "lat": 40.7271, "lon": -74.0776, "region": "Urban", "facility_size": 140, "nurse_ratio": 3, "specialist_avail": 7},
    # Suburban – New Jersey / Northeast
    {"name": "Morristown Medical Center",   "city": "Morristown, NJ",    "lat": 40.7965, "lon": -74.4810, "region": "Urban", "facility_size": 185, "nurse_ratio": 2, "specialist_avail": 9},
    {"name": "Saint Clare's Health",        "city": "Dover, NJ",         "lat": 40.8837, "lon": -74.5574, "region": "Urban", "facility_size": 120, "nurse_ratio": 3, "specialist_avail": 6},
    {"name": "Chilton Medical Center",      "city": "Pompton Plains, NJ","lat": 40.9807, "lon": -74.2987, "region": "Urban", "facility_size": 110, "nurse_ratio": 3, "specialist_avail": 5},
    {"name": "Hackettstown Medical Center", "city": "Hackettstown, NJ",  "lat": 40.8540, "lon": -74.8289, "region": "Rural", "facility_size": 65,  "nurse_ratio": 4, "specialist_avail": 3},
    {"name": "Overlook Medical Center",     "city": "Summit, NJ",        "lat": 40.7157, "lon": -74.3580, "region": "Urban", "facility_size": 160, "nurse_ratio": 2, "specialist_avail": 8},
    {"name": "Newton Medical Center",       "city": "Newton, NJ",        "lat": 41.0573, "lon": -74.7527, "region": "Rural", "facility_size": 75,  "nurse_ratio": 4, "specialist_avail": 3},
    # Rural
    {"name": "Valley Community Hospital",   "city": "Harrisburg, PA",    "lat": 40.2732, "lon": -76.8867, "region": "Rural", "facility_size": 45,  "nurse_ratio": 4, "specialist_avail": 3},
    {"name": "Blue Ridge ER",               "city": "Roanoke, VA",       "lat": 37.2710, "lon": -79.9414, "region": "Rural", "facility_size": 60,  "nurse_ratio": 4, "specialist_avail": 2},
    {"name": "Prairie Health Center",       "city": "Wichita, KS",       "lat": 37.6872, "lon": -97.3301, "region": "Rural", "facility_size": 40,  "nurse_ratio": 5, "specialist_avail": 2},
    {"name": "Appalachian Regional",        "city": "Lexington, KY",     "lat": 38.0406, "lon": -84.5037, "region": "Rural", "facility_size": 55,  "nurse_ratio": 4, "specialist_avail": 3},
    {"name": "Cascade Valley Medical",      "city": "Salem, OR",         "lat": 44.9429, "lon": -123.0351,"region": "Rural", "facility_size": 50,  "nurse_ratio": 4, "specialist_avail": 2},
    {"name": "Great Plains Community ER",   "city": "Fargo, ND",         "lat": 46.8772, "lon": -96.7898, "region": "Rural", "facility_size": 35,  "nurse_ratio": 5, "specialist_avail": 1},
    {"name": "Ozark Hills Hospital",        "city": "Springfield, MO",   "lat": 37.2090, "lon": -93.2923, "region": "Rural", "facility_size": 65,  "nurse_ratio": 4, "specialist_avail": 3},
])

# ── Helpers: time context ──────────────────────────────────────────────────────
def current_time_of_day():
    h = datetime.now().hour
    if 5 <= h < 10:  return "Early Morning"
    if 10 <= h < 12: return "Late Morning"
    if 12 <= h < 17: return "Afternoon"
    if 17 <= h < 21: return "Evening"
    return "Night"

def current_season():
    m = datetime.now().month
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    return "Fall"

def current_day_of_week():
    return datetime.now().strftime("%A")

# ── Haversine distance (miles) ─────────────────────────────────────────────────
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ── Travel time via OSRM (free, no API key) ────────────────────────────────────
def get_travel_minutes(lat1, lon1, lat2, lon2, region="Urban"):
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?overview=false"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return round(data["routes"][0]["duration"] / 60, 1)
    except Exception:
        pass
    dist = haversine_miles(lat1, lon1, lat2, lon2)
    mph = 25 if region == "Urban" else 50
    return round((dist / mph) * 60 + 5, 1)

# ── Geocode address ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="ertime_app_aai595")
        loc = geolocator.geocode(address, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except Exception:
        pass
    return None, None, None

# ── Train model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("cleaned_data.csv")
    feature_cols = [
        "Region", "Day of Week", "Season", "Time of Day", "Urgency Level",
        "Nurse-to-Patient Ratio", "Specialist Availability", "Facility Size (Beds)"
    ]
    X = df[feature_cols].copy()
    y = df["Total Wait Time (min)"].copy()

    label_encoders = {}
    for col in ["Region", "Day of Week", "Season", "Time of Day", "Urgency Level"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(math.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }
    return model, scaler, label_encoders, metrics

model, scaler, label_encoders, metrics = load_model()

# ── Predict wait time for a single hospital row ────────────────────────────────
def predict_wait(hospital, urgency, time_of_day, day_of_week, season):
    row = {
        "Region":                  label_encoders["Region"].transform([hospital["region"]])[0],
        "Day of Week":             label_encoders["Day of Week"].transform([day_of_week])[0],
        "Season":                  label_encoders["Season"].transform([season])[0],
        "Time of Day":             label_encoders["Time of Day"].transform([time_of_day])[0],
        "Urgency Level":           label_encoders["Urgency Level"].transform([urgency])[0],
        "Nurse-to-Patient Ratio":  hospital["nurse_ratio"],
        "Specialist Availability": hospital["specialist_avail"],
        "Facility Size (Beds)":    hospital["facility_size"],
    }
    X_in = pd.DataFrame([row])
    return round(model.predict(scaler.transform(X_in))[0], 1)

# ── Wait-time color helper ─────────────────────────────────────────────────────
def wait_color(minutes):
    if minutes < 45:   return "green"
    if minutes < 90:   return "orange"
    return "red"

def fmt_time(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m} min"

# ── App header ─────────────────────────────────────────────────────────────────
st.title("🏥 ERTime — Emergency Room Wait Time Predictor")
st.caption(
    "AAI 595 · Random Forest  |  "
    "Michael Lugo · Amoy Mowatt · Ardit Cana · Wesley Nabo"
)
st.divider()

tab1, tab2 = st.tabs(["📍 Find ER Near Me", "🔬 Manual Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Location-based ER finder
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Find the Fastest ER From Your Location")
    st.caption(
        "Enter your address or ZIP code. We'll predict wait times at nearby ERs "
        "and rank them by **total time to care** (travel + predicted wait)."
    )

    col_addr, col_urg = st.columns([3, 1])
    with col_addr:
        address_input = st.text_input(
            "Your current location",
            placeholder="e.g. 350 5th Ave, New York, NY  or  10001",
        )
    with col_urg:
        urgency_loc = st.selectbox(
            "Urgency Level",
            ["Low", "Medium", "High", "Critical"],
            index=1,
            key="urgency_loc",
        )

    n_results = st.slider("Number of nearest ERs to show", 3, 10, 5)

    if address_input:
        with st.spinner("Geocoding your address..."):
            user_lat, user_lon, resolved = geocode_address(address_input)

        if user_lat is None:
            st.error("Could not find that address. Try adding a city or state.")
        else:
            st.success(f"Location resolved: {resolved}")

            tod    = current_time_of_day()
            season = current_season()
            dow    = current_day_of_week()
            st.caption(
                f"Using current time context: **{dow}** · **{tod}** · **{season}**"
            )

            rows = []
            progress = st.progress(0, text="Calculating routes & predictions...")
            for i, (_, hosp) in enumerate(HOSPITALS.iterrows()):
                dist   = haversine_miles(user_lat, user_lon, hosp["lat"], hosp["lon"])
                travel = get_travel_minutes(user_lat, user_lon, hosp["lat"], hosp["lon"], hosp["region"])
                wait   = predict_wait(hosp, urgency_loc, tod, dow, season)
                rows.append({
                    "name":        hosp["name"],
                    "city":        hosp["city"],
                    "region":      hosp["region"],
                    "lat":         hosp["lat"],
                    "lon":         hosp["lon"],
                    "dist_mi":     round(dist, 1),
                    "travel_min":  travel,
                    "wait_min":    wait,
                    "total_min":   round(travel + wait, 1),
                    "beds":        hosp["facility_size"],
                    "nurse_ratio": hosp["nurse_ratio"],
                })
                progress.progress((i + 1) / len(HOSPITALS))
            progress.empty()

            # Critical urgency: prioritize fastest arrival (travel time);
            # all others: minimize total time to care (travel + predicted wait).
            sort_col = "travel_min" if urgency_loc == "Critical" else "total_min"
            results = (
                pd.DataFrame(rows)
                .sort_values("dist_mi")
                .head(n_results)
                .sort_values(sort_col)
                .reset_index(drop=True)
            )
            best = results.iloc[0]

            sort_label = "fastest drive time" if urgency_loc == "Critical" else "lowest total time to care"
            st.success(
                f"**Recommended:** {best['name']} ({best['city']})  \n"
                f"Drive: **{fmt_time(best['travel_min'])}** · "
                f"Wait: **{fmt_time(best['wait_min'])}** · "
                f"Total: **{fmt_time(best['total_min'])}**  \n"
                f"*Ranked by {sort_label}*"
            )

            map_col, table_col = st.columns([3, 2])

            with map_col:
                m = folium.Map(location=[user_lat, user_lon], zoom_start=9)

                folium.Marker(
                    location=[user_lat, user_lon],
                    tooltip="Your Location",
                    icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                ).add_to(m)

                for rank, row in results.iterrows():
                    color = wait_color(row["wait_min"])
                    label = "★ " if rank == 0 else f"{rank+1}. "
                    popup_html = f"""
                    <b>{label}{row['name']}</b><br>
                    {row['city']} ({row['region']})<br>
                    <hr style='margin:4px 0'>
                    🚗 Drive: <b>{fmt_time(row['travel_min'])}</b><br>
                    ⏱ Wait:  <b>{fmt_time(row['wait_min'])}</b><br>
                    📊 Total: <b>{fmt_time(row['total_min'])}</b><br>
                    🛏 Beds: {row['beds']}
                    """
                    folium.Marker(
                        location=[row["lat"], row["lon"]],
                        tooltip=f"{label}{row['name']}",
                        popup=folium.Popup(popup_html, max_width=220),
                        icon=folium.Icon(color=color, icon="plus", prefix="fa"),
                    ).add_to(m)

                    folium.PolyLine(
                        locations=[[user_lat, user_lon], [row["lat"], row["lon"]]],
                        color=color,
                        weight=1.5,
                        opacity=0.5,
                        dash_array="6",
                    ).add_to(m)

                st_folium(m, height=420, use_container_width=True)
                st.caption(
                    "🟢 < 45 min wait &nbsp;|&nbsp; 🟠 45–90 min &nbsp;|&nbsp; 🔴 > 90 min"
                )

            with table_col:
                st.markdown("#### Ranked by Total Time to Care")
                for rank, row in results.iterrows():
                    medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"
                    with st.container(border=True):
                        st.markdown(
                            f"**{medal} {row['name']}**  \n"
                            f"*{row['city']} · {row['region']}*"
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Drive",  fmt_time(row["travel_min"]))
                        c2.metric("Wait",   fmt_time(row["wait_min"]))
                        c3.metric("Total",  fmt_time(row["total_min"]))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Manual Wait Time Prediction")
    st.caption("Enter patient and facility details to predict ER wait time.")

    col1, col2 = st.columns(2)

    with col1:
        urgency  = st.selectbox("Urgency Level", ["Low", "Medium", "High", "Critical"], index=1)
        region   = st.selectbox("Region", ["Rural", "Urban"])
        tod_man  = st.selectbox(
            "Time of Day",
            ["Early Morning", "Late Morning", "Afternoon", "Evening", "Night"],
            index=["Early Morning","Late Morning","Afternoon","Evening","Night"]
                .index(current_time_of_day()),
        )
        dow_man  = st.selectbox(
            "Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            index=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                .index(current_day_of_week()),
        )
        seas_man = st.selectbox(
            "Season",
            ["Spring","Summer","Fall","Winter"],
            index=["Spring","Summer","Fall","Winter"].index(current_season()),
        )

    with col2:
        nurse_ratio    = st.slider("Nurse-to-Patient Ratio",  1, 5,   3)
        specialist_av  = st.slider("Specialist Availability", 0, 10,  5)
        facility_sz    = st.slider("Facility Size (Beds)",    10, 200, 100, step=5)

    st.divider()

    if st.button("Predict Wait Time", type="primary", use_container_width=True):
        hosp_manual = {
            "region": region, "nurse_ratio": nurse_ratio,
            "specialist_avail": specialist_av, "facility_size": facility_sz,
        }
        pred = predict_wait(hosp_manual, urgency, tod_man, dow_man, seas_man)
        h, m = divmod(int(pred), 60)
        time_str = f"{h}h {m}m" if h else f"{m} minutes"

        icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        st.metric(
            label=f"{icons[urgency]} Estimated Wait ({urgency} urgency)",
            value=time_str,
            delta=f"±{metrics['mae']} min (model MAE)",
            delta_color="off",
        )

        if pred < 45:
            st.success("Short wait — patient should be seen quickly.")
        elif pred < 90:
            st.warning("Moderate wait — typical for current conditions.")
        else:
            st.error("Long wait — consider alternative facilities if non-critical.")

    with st.expander("Model performance"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Test R²",   metrics["r2"])
        c2.metric("Test MAE",  f"{metrics['mae']} min")
        c3.metric("Test RMSE", f"{metrics['rmse']} min")
        st.caption(
            "Linear Regression baseline: R² 0.5348, MAE 34.81 min — "
            "Random Forest reduces error by ~64%."
        )

st.divider()
st.caption(
    "Model trained on `cleaned_data.csv` · "
    "Features reflect information available at patient arrival (no data leakage) · "
    "Travel times via OSRM routing API"
)
