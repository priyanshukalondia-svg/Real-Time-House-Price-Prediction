import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Real-Time House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "house_price_pipeline.joblib"

model = joblib.load(MODEL_PATH)

# ==========================================================
# TITLE
# ==========================================================

st.title("🏠 Real-Time House Price Prediction System")

st.markdown("""
Predict house prices using a **Gradient Boosting Regressor**
trained on the Ames Housing Dataset.

---
""")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🏠 House Price Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.success("R² Score : 0.9343")

st.sidebar.info("MAE : $13,482")

st.sidebar.info("RMSE : $22,945")

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.write("""
This application predicts residential house prices using Machine Learning.

Model Used:
- Gradient Boosting Regressor

Techniques Used:
- Feature Engineering
- Pipeline
- ColumnTransformer
- Hyperparameter Tuning
- Cross Validation
""")

# ==========================================================
# DEFAULT VALUES
# ==========================================================

numeric_defaults = {
    'MS SubClass':50,
    'Lot Frontage':68,
    'Lot Area':9436.5,
    'Overall Qual':6,
    'Overall Cond':5,
    'Year Built':1973,
    'Year Remod/Add':1993,
    'Mas Vnr Area':0,
    'BsmtFin SF 1':370,
    'BsmtFin SF 2':0,
    'Bsmt Unf SF':465.5,
    'Total Bsmt SF':990,
    '1st Flr SF':1084,
    '2nd Flr SF':0,
    'Low Qual Fin SF':0,
    'Gr Liv Area':1442,
    'Bsmt Full Bath':0,
    'Bsmt Half Bath':0,
    'Full Bath':2,
    'Half Bath':0,
    'Bedroom AbvGr':3,
    'Kitchen AbvGr':1,
    'TotRms AbvGrd':6,
    'Fireplaces':1,
    'Garage Yr Blt':1979,
    'Garage Cars':2,
    'Garage Area':480,
    'Wood Deck SF':0,
    'Open Porch SF':27,
    'Enclosed Porch':0,
    '3Ssn Porch':0,
    'Screen Porch':0,
    'Pool Area':0,
    'Misc Val':0,
    'Mo Sold':6,
    'Yr Sold':2008
}

categorical_defaults = {
    'MS Zoning':'RL',
    'Street':'Pave',
    'Alley':'None',
    'Lot Shape':'Reg',
    'Land Contour':'Lvl',
    'Utilities':'AllPub',
    'Lot Config':'Inside',
    'Land Slope':'Gtl',
    'Neighborhood':'NAmes',
    'Condition 1':'Norm',
    'Condition 2':'Norm',
    'Bldg Type':'1Fam',
    'House Style':'1Story',
    'Roof Style':'Gable',
    'Roof Matl':'CompShg',
    'Exterior 1st':'VinylSd',
    'Exterior 2nd':'VinylSd',
    'Mas Vnr Type':'None',
    'Exter Qual':'TA',
    'Exter Cond':'TA',
    'Foundation':'PConc',
    'Bsmt Qual':'TA',
    'Bsmt Cond':'TA',
    'Bsmt Exposure':'No',
    'BsmtFin Type 1':'GLQ',
    'BsmtFin Type 2':'Unf',
    'Heating':'GasA',
    'Heating QC':'Ex',
    'Central Air':'Y',
    'Electrical':'SBrkr',
    'Kitchen Qual':'TA',
    'Functional':'Typ',
    'Fireplace Qu':'None',
    'Garage Type':'Attchd',
    'Garage Finish':'Unf',
    'Garage Qual':'TA',
    'Garage Cond':'TA',
    'Paved Drive':'Y',
    'Pool QC':'None',
    'Fence':'None',
    'Misc Feature':'None',
    'Sale Type':'WD ',
    'Sale Condition':'Normal'
}

# ==========================================================
# HELPER
# ==========================================================

def format_currency(price):
    return "${:,.0f}".format(price)

# ==========================================================
# USER INPUT
# ==========================================================

left, right = st.columns(2)

with left:

    st.subheader("Property Information")

    overall_qual = st.slider(
        "Overall Quality",
        1,
        10,
        6
    )

    overall_cond = st.slider(
        "Overall Condition",
        1,
        10,
        5
    )

    lot_area = st.number_input(
        "Lot Area",
        value=9500
    )

    gr_liv_area = st.number_input(
        "Ground Living Area",
        value=1500
    )

    garage_cars = st.slider(
        "Garage Capacity",
        0,
        5,
        2
    )

    garage_area = st.number_input(
        "Garage Area",
        value=480
    )

    fireplaces = st.slider(
        "Fireplaces",
        0,
        4,
        1
    )

with right:

    st.subheader("Construction")

    year_built = st.number_input(
        "Year Built",
        1870,
        2025,
        2005
    )

    year_remod = st.number_input(
        "Year Remodeled",
        1950,
        2025,
        2008
    )

    total_bsmt = st.number_input(
        "Basement Area",
        value=1000
    )

    first_floor = st.number_input(
        "1st Floor Area",
        value=1200
    )

    second_floor = st.number_input(
        "2nd Floor Area",
        value=300
    )

    full_bath = st.slider(
        "Full Bathrooms",
        0,
        5,
        2
    )

    half_bath = st.slider(
        "Half Bathrooms",
        0,
        3,
        1
    )

predict = st.button(
    "Predict House Price",
    use_container_width=True
)
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    try:

        # Start with defaults
        data = {}

        data.update(numeric_defaults)
        data.update(categorical_defaults)

        # -----------------------------
        # User Inputs
        # -----------------------------

        data["Overall Qual"] = overall_qual
        data["Overall Cond"] = overall_cond
        data["Lot Area"] = lot_area
        data["Gr Liv Area"] = gr_liv_area
        data["Garage Cars"] = garage_cars
        data["Garage Area"] = garage_area
        data["Fireplaces"] = fireplaces

        data["Year Built"] = year_built
        data["Year Remod/Add"] = year_remod

        data["Total Bsmt SF"] = total_bsmt
        data["1st Flr SF"] = first_floor
        data["2nd Flr SF"] = second_floor

        data["Full Bath"] = full_bath
        data["Half Bath"] = half_bath

        # -----------------------------
        # Feature Engineering
        # -----------------------------

        data["HouseAge"] = (
            data["Yr Sold"] -
            data["Year Built"]
        )

        data["RemodAge"] = (
            data["Yr Sold"] -
            data["Year Remod/Add"]
        )

        data["TotalBathrooms"] = (
            data["Full Bath"]
            + 0.5 * data["Half Bath"]
            + data["Bsmt Full Bath"]
            + 0.5 * data["Bsmt Half Bath"]
        )

        data["TotalSF"] = (
            data["Total Bsmt SF"]
            + data["1st Flr SF"]
            + data["2nd Flr SF"]
        )

        data["TotalPorchSF"] = (
            data["Wood Deck SF"]
            + data["Open Porch SF"]
            + data["Enclosed Porch"]
            + data["3Ssn Porch"]
            + data["Screen Porch"]
        )

        # -----------------------------
        # Convert to DataFrame
        # -----------------------------

        input_df = pd.DataFrame([data])

        # -----------------------------
        # Predict
        # -----------------------------

        prediction = model.predict(input_df)[0]

        # -----------------------------
        # Output
        # -----------------------------

        st.markdown("---")

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="🏠 Estimated House Price",
            value=format_currency(prediction)
        )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Quality",
                overall_qual
            )

        with col2:
            st.metric(
                "Living Area",
                f"{gr_liv_area:,} sq.ft"
            )

        with col3:
            st.metric(
                "Garage",
                garage_cars
            )

        st.markdown("---")

        st.subheader("Model Used")

        st.write("✅ Gradient Boosting Regressor")

        st.write("R² Score : **0.9343**")

        st.write("MAE : **$13,482**")

        st.write("RMSE : **$22,945**")

        st.markdown("---")

        with st.expander("View Submitted Features"):

            st.dataframe(
                input_df.T,
                use_container_width=True
            )

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Developed by Priyanshu Kalondia | "
    "Machine Learning Portfolio Project | "
    "Real-Time House Price Prediction System"
)