from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
FEATURE_LABELS = {
    "MedInc": "Median income",
    "HouseAge": "House age",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Population",
    "AveOccup": "Average occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}
PRESETS = {
    "Los Angeles": (34.05, -118.25),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
    "Custom coordinates": None,
}
AVG_PRICE = 206_856  # average median house value in the dataset

# Update these numbers if you retrain and get different results
RESULTS = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [0.7456, 0.5059],
        "R²": [0.5758, 0.8047],
    }
)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
    .hero p { color: #dbe7ff; margin: 0.4rem 0 0 0; font-size: 1.05rem; }
    .price-card {
        background: linear-gradient(135deg, #0f766e, #14b8a6);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; margin-bottom: 1rem;
    }
    .price-card .label {
        font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.9;
    }
    .price-card .value { font-size: 3.4rem; font-weight: 800; line-height: 1.2; }
    .price-card .range { font-size: 1rem; opacity: 0.95; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def predict_price(model, scaler, values):
    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = scaler.transform(X)
    price = model.predict(X_scaled)[0] * 100_000  # target is in $100k units
    low = high = None
    if hasattr(model, "estimators_"):  # Random Forest: use the spread across trees
        trees = np.array([t.predict(X_scaled)[0] for t in model.estimators_]) * 100_000
        low, high = np.percentile(trees, [10, 90])
    return price, low, high


st.markdown(
    """
    <div class="hero">
        <h1>🏠 California House Price Predictor</h1>
        <p>A machine learning model that estimates median house value from census data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python -m src.train` first.")
    st.stop()

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("🔧 House details")

    st.subheader("Block characteristics")
    med_inc = st.slider("Median income (×$10,000)", 0.5, 15.0, 3.5, 0.1)
    house_age = st.slider("House age (years)", 1, 52, 28)
    ave_rooms = st.slider("Average rooms", 1.0, 15.0, 5.4, 0.1)
    ave_bedrms = st.slider("Average bedrooms", 0.5, 5.0, 1.1, 0.1)
    population = st.number_input("Population", 3, 35000, 1400, step=50)
    ave_occup = st.slider("Average occupancy", 1.0, 10.0, 3.0, 0.1)

    st.subheader("Location")
    place = st.selectbox("Choose a place", list(PRESETS))
    if PRESETS[place]:
        latitude, longitude = PRESETS[place]
        st.caption(f"Latitude {latitude}, Longitude {longitude}")
    else:
        latitude = st.number_input("Latitude", 32.5, 42.0, 34.05, format="%.2f")
        longitude = st.number_input("Longitude", -124.4, -114.3, -118.25, format="%.2f")

    clicked = st.button("Predict Price", type="primary", use_container_width=True)

# Predict on the first load (so the page isn't empty) and on every button click
if clicked or "result" not in st.session_state:
    values = {
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
        "Latitude": latitude, "Longitude": longitude,
    }
    price, low, high = predict_price(model, scaler, values)
    st.session_state["result"] = {
        "price": price, "low": low, "high": high, "lat": latitude, "lon": longitude,
    }

res = st.session_state["result"]
model_name = "Random Forest" if hasattr(model, "estimators_") else "Linear Regression"

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📍 Prediction", "📊 Model performance", "ℹ️ About"])

with tab1:
    left, right = st.columns([3, 2])

    with left:
        range_html = ""
        if res["low"] is not None:
            range_html = (
                f'<div class="range">Likely range: ${res["low"]:,.0f} – ${res["high"]:,.0f}</div>'
            )
        st.markdown(
            f"""
            <div class="price-card">
                <div class="label">Estimated median house value</div>
                <div class="value">${res["price"]:,.0f}</div>
                {range_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        diff = res["price"] - AVG_PRICE
        c1, c2, c3 = st.columns(3)
        c1.metric("California average", f"${AVG_PRICE:,.0f}")
        c2.metric("This estimate", f"${res['price']:,.0f}", delta=f"{diff / AVG_PRICE:+.0%} vs average")
        c3.metric("Model used", model_name)

    with right:
        st.markdown("**📍 Location**")
        st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}), zoom=8, size=3000)

    st.caption("Change the inputs in the sidebar and click **Predict Price** to update.")

with tab2:
    st.subheader("Model comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(RESULTS, hide_index=True, use_container_width=True)
        st.caption("RMSE is in units of $100,000 (lower is better). R² closer to 1 is better.")
    with c2:
        st.bar_chart(RESULTS.set_index("Model")["R²"])

    st.subheader("What drives the price?")
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=[FEATURE_LABELS[f] for f in FEATURES],
        ).sort_values()
        st.bar_chart(importance, horizontal=True)
        st.caption("Longer bars mean the feature has more influence on the predicted price.")
    else:
        st.info("Feature importance is available when the Random Forest model is used.")

with tab3:
    st.subheader("About this project")
    st.markdown(
        """
        - **Data:** California Housing dataset (1990 US census), 20,640 block groups, 8 features
        - **Models:** Linear Regression baseline and Random Forest, compared by RMSE and R²
        - **Pipeline:** train/test split, StandardScaler, best model saved with joblib
        - **Tech:** Python, Scikit-Learn, Pandas, Streamlit

        **Limitations:** the data is from 1990, so prices are much lower than today's market,
        and each prediction describes a census block group, not a single house.
        """
    )