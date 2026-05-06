# =========================================================
# Streamlit App - Telco Customer Churn Prediction
# Model file assumption: best_pipe_model_.pkl
# Run command: streamlit run app_streamlit_churn.py
# =========================================================

import os
import pickle
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import streamlit as st


# =============================
# 1. Page Configuration
# =============================
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="wide"
)


# =============================
# 2. Required Model Features
#    These names must match the training columns exactly.
# =============================
FEATURE_COLUMNS: List[str] = [
    "avgmonthlygbdownload",
    "avgmonthlylongdistancecharges",
    "contract",
    "dependents",
    "deviceprotection",
    "internetservice",
    "multiplelines",
    "numberofreferrals",
    "onlinebackup",
    "onlinesecurity",
    "paperlessbilling",
    "partner",
    "paymentmethod",
    "phoneservice",
    "population",
    "referredafriend",
    "satisfactionscore",
    "seniorcitizen",
    "streamingmusic",
    "streamingmovies",
    "streamingtv",
    "techsupport",
    "totalextradatacharges",
    "totalrefunds",
    "totalcharges",
    "unlimiteddata",
    "gender",
]

# Categorical values from categorical_feature_categories.csv
CATEGORY_OPTIONS: Dict[str, List[Any]] = {
    "contract": ["One year", "Month-to-month", "Two year"],
    "dependents": ["Yes", "No"],
    "deviceprotection": ["No", "Yes", "No internet service"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "multiplelines": ["No", "Yes", "No phone service"],
    "onlinebackup": ["Yes", "No", "No internet service"],
    "onlinesecurity": ["No", "Yes", "No internet service"],
    "paperlessbilling": ["Yes", "No"],
    "partner": ["Yes", "No"],
    "paymentmethod": [
        "Mailed check",
        "Electronic check",
        "Credit card (automatic)",
        "Bank transfer (automatic)",
    ],
    "phoneservice": ["Yes", "No"],
    "seniorcitizen": ["No", "Yes"],
    "streamingmovies": ["No", "Yes", "No internet service"],
    "streamingtv": ["Yes", "No", "No internet service"],
    "techsupport": ["Yes", "No", "No internet service"],
    "gender": ["Female", "Male"],
}

# Numeric/binary columns based on the model specification
NUMERIC_COLUMNS: List[str] = [
    "avgmonthlygbdownload",
    "avgmonthlylongdistancecharges",
    "numberofreferrals",
    "population",
    "referredafriend",
    "satisfactionscore",
    "streamingmusic",
    "totalextradatacharges",
    "totalrefunds",
    "totalcharges",
    "unlimiteddata",
]


# =============================
# 3. Helper Functions
# =============================
@st.cache_resource
def load_model(model_path: str):
    """Load the saved sklearn/imblearn pipeline model."""
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def make_prediction(model, input_df: pd.DataFrame, threshold: float):
    """Return probability and churn prediction based on selected threshold."""
    # Default probability if model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(input_df)

        # Try to identify positive class correctly: 1 or Yes.
        classes = None
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            if hasattr(estimator, "classes_"):
                classes = list(estimator.classes_)

        if classes is not None:
            if 1 in classes:
                positive_index = classes.index(1)
            elif "Yes" in classes:
                positive_index = classes.index("Yes")
            elif "Churn" in classes:
                positive_index = classes.index("Churn")
            else:
                positive_index = 1 if proba_all.shape[1] > 1 else 0
        else:
            positive_index = 1 if proba_all.shape[1] > 1 else 0

        churn_probability = float(proba_all[0, positive_index])
        prediction = 1 if churn_probability >= threshold else 0
        return prediction, churn_probability

    # Fallback when model has no predict_proba
    prediction_raw = model.predict(input_df)[0]
    prediction = 1 if prediction_raw in [1, "Yes", "Churn"] else 0
    return prediction, np.nan


def yes_no_to_binary(value: str) -> int:
    """Convert Yes/No choice into 1/0 for binary columns trained as numeric."""
    return 1 if value == "Yes" else 0


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns exist and are ordered exactly like training data."""
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Keep only required columns and force the same column order as training.
    df = df[FEATURE_COLUMNS].copy()

    # Convert numeric columns safely.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================
# 4. Sidebar
# =============================
st.sidebar.title("⚙️ Model Settings")
model_path = "best_pipe_model_.pkl"

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
    help="Jika probability churn >= threshold, customer diprediksi Churn."
)

model = load_model(model_path)


# =============================
# 5. Main Page
# =============================
st.title("📉 Customer Churn Prediction App")
st.write(
    "Aplikasi ini digunakan untuk memprediksi apakah customer berpotensi churn "
    "berdasarkan pipeline model yang sudah dilatih."
)

if model is None:
    st.error(
        f"Model `{model_path}` belum ditemukan. Pastikan file model ada di folder yang sama dengan app Streamlit ini."
    )
    st.stop()
else:
    st.success(f"Model `{model_path}` berhasil dimuat.")


# =============================
# 6. Input Mode
# =============================
input_mode = st.radio(
    "Pilih metode input",
    ["Single Customer", "Batch Prediction via CSV"],
    horizontal=True
)


# =============================
# 7. Single Customer Input
# =============================
if input_mode == "Single Customer":
    st.subheader("Input Data Customer")

    with st.form("single_prediction_form"):
        # -------------------------
        # A. Customer Profile
        # -------------------------
        st.markdown("### 1. Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
            seniorcitizen = st.selectbox("Senior Citizen", CATEGORY_OPTIONS["seniorcitizen"])
            partner = st.selectbox("Partner", CATEGORY_OPTIONS["partner"])

        with col2:
            dependents = st.selectbox("Dependents", CATEGORY_OPTIONS["dependents"])
            referredafriend_choice = st.selectbox("Referred a Friend", ["No", "Yes"])
            numberofreferrals = st.number_input("Number of Referrals", min_value=0, value=0, step=1)

        with col3:
            population = st.number_input("Population", min_value=0, value=4498, step=1)
            satisfactionscore = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

        # -------------------------
        # B. Account & Contract
        # -------------------------
        st.markdown("### 2. Account & Contract")
        col1, col2, col3 = st.columns(3)

        with col1:
            contract = st.selectbox("Contract", CATEGORY_OPTIONS["contract"])
        with col2:
            paymentmethod = st.selectbox("Payment Method", CATEGORY_OPTIONS["paymentmethod"])
        with col3:
            paperlessbilling = st.selectbox("Paperless Billing", CATEGORY_OPTIONS["paperlessbilling"])

        # -------------------------
        # C. Phone Service
        # -------------------------
        st.markdown("### 3. Phone Service")
        col1, col2, col3 = st.columns(3)

        with col1:
            phoneservice = st.selectbox("Phone Service", CATEGORY_OPTIONS["phoneservice"])
        with col2:
            multiplelines = st.selectbox("Multiple Lines", CATEGORY_OPTIONS["multiplelines"])
        with col3:
            avgmonthlylongdistancecharges = st.number_input(
                "Avg Monthly Long Distance Charges",
                min_value=0.0,
                value=42.39,
                step=0.01,
                format="%.2f"
            )

        # -------------------------
        # D. Internet & Data Usage
        # -------------------------
        st.markdown("### 4. Internet & Data Usage")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            internetservice = st.selectbox("Internet Service", CATEGORY_OPTIONS["internetservice"])
        with col2:
            unlimiteddata_choice = st.selectbox("Unlimited Data", ["No", "Yes"], index=1)
        with col3:
            avgmonthlygbdownload = st.number_input("Avg Monthly GB Download", min_value=0, value=16, step=1)
        with col4:
            totalextradatacharges = st.number_input(
                "Total Extra Data Charges",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.2f"
            )

        # -------------------------
        # E. Add-on Services
        # -------------------------
        st.markdown("### 5. Add-on Services")
        col1, col2, col3 = st.columns(3)

        with col1:
            onlinebackup = st.selectbox("Online Backup", CATEGORY_OPTIONS["onlinebackup"])
            onlinesecurity = st.selectbox("Online Security", CATEGORY_OPTIONS["onlinesecurity"])
        with col2:
            deviceprotection = st.selectbox("Device Protection", CATEGORY_OPTIONS["deviceprotection"])
            techsupport = st.selectbox("Tech Support", CATEGORY_OPTIONS["techsupport"])
        with col3:
            streamingtv = st.selectbox("Streaming TV", CATEGORY_OPTIONS["streamingtv"])
            streamingmovies = st.selectbox("Streaming Movies", CATEGORY_OPTIONS["streamingmovies"])
            streamingmusic_choice = st.selectbox("Streaming Music", ["No", "Yes"])

        # -------------------------
        # F. Billing Amount
        # -------------------------
        st.markdown("### 6. Billing Amount")
        col1, col2 = st.columns(2)

        with col1:
            totalcharges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=593.30,
                step=0.01,
                format="%.2f"
            )
        with col2:
            totalrefunds = st.number_input(
                "Total Refunds",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f"
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = {
            "avgmonthlygbdownload": avgmonthlygbdownload,
            "avgmonthlylongdistancecharges": avgmonthlylongdistancecharges,
            "contract": contract,
            "dependents": dependents,
            "deviceprotection": deviceprotection,
            "internetservice": internetservice,
            "multiplelines": multiplelines,
            "numberofreferrals": numberofreferrals,
            "onlinebackup": onlinebackup,
            "onlinesecurity": onlinesecurity,
            "paperlessbilling": paperlessbilling,
            "partner": partner,
            "paymentmethod": paymentmethod,
            "phoneservice": phoneservice,
            "population": population,
            "referredafriend": yes_no_to_binary(referredafriend_choice),
            "satisfactionscore": satisfactionscore,
            "seniorcitizen": seniorcitizen,
            "streamingmusic": yes_no_to_binary(streamingmusic_choice),
            "streamingmovies": streamingmovies,
            "streamingtv": streamingtv,
            "techsupport": techsupport,
            "totalextradatacharges": totalextradatacharges,
            "totalrefunds": totalrefunds,
            "totalcharges": totalcharges,
            "unlimiteddata": yes_no_to_binary(unlimiteddata_choice),
            "gender": gender,
        }

        input_df = pd.DataFrame([input_data])
        input_df = validate_input(input_df)

        prediction, churn_probability = make_prediction(model, input_df, threshold)

        st.markdown("---")
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("Prediction: Customer berpotensi CHURN")
            else:
                st.success("Prediction: Customer berpotensi TIDAK CHURN")

        with result_col2:
            if not np.isnan(churn_probability):
                st.metric("Churn Probability", f"{churn_probability:.2%}")
            else:
                st.info("Model tidak menyediakan probability score.")

        with st.expander("Lihat data input yang dikirim ke model"):
            st.dataframe(input_df, use_container_width=True)


# =============================
# 8. Batch Prediction
# =============================
else:
    st.subheader("Batch Prediction via CSV")
    st.write(
        "Upload file CSV dengan kolom yang sama seperti data training. "
        "Kolom harus memakai nama fitur yang sama persis."
    )

    with st.expander("Daftar kolom yang wajib ada"):
        st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_input = validate_input(batch_df)

            if hasattr(model, "predict_proba"):
                proba_all = model.predict_proba(batch_input)

                classes = None
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                elif hasattr(model, "named_steps") and "model" in model.named_steps:
                    estimator = model.named_steps["model"]
                    if hasattr(estimator, "classes_"):
                        classes = list(estimator.classes_)

                if classes is not None and 1 in classes:
                    positive_index = classes.index(1)
                elif classes is not None and "Yes" in classes:
                    positive_index = classes.index("Yes")
                else:
                    positive_index = 1 if proba_all.shape[1] > 1 else 0

                churn_probability = proba_all[:, positive_index]
                prediction = (churn_probability >= threshold).astype(int)
            else:
                raw_prediction = model.predict(batch_input)
                prediction = [1 if pred in [1, "Yes", "Churn"] else 0 for pred in raw_prediction]
                churn_probability = [np.nan] * len(batch_input)

            result_df = batch_df.copy()
            result_df["churn_probability"] = churn_probability
            result_df["prediction"] = prediction
            result_df["prediction_label"] = result_df["prediction"].map({1: "Churn", 0: "Not Churn"})

            st.success("Batch prediction berhasil dibuat.")
            st.dataframe(result_df, use_container_width=True)

            csv_result = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Result",
                data=csv_result,
                file_name="churn_prediction_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Terjadi error saat memproses file: {e}")


