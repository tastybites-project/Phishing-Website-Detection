
import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# STEP 1: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Phishing Website Detection System")

st.write(
    "Machine Learning based website classification system "
    "to identify potentially phishing websites."
)

st.divider()


# ==========================================
# STEP 2: LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/best_phishing_model.pkl"
TRAIN_PATH = "dataset/X_train.csv"

if not os.path.exists(MODEL_PATH):
    st.error("Best model not found. Run hyperparameter_tuning.py first.")
    st.stop()

if not os.path.exists(TRAIN_PATH):
    st.error("Training dataset not found. Run split_data.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Load training feature information
X_train = pd.read_csv(TRAIN_PATH)

feature_columns = X_train.columns.tolist()


# ==========================================
# STEP 3: DISPLAY PROJECT INFORMATION
# ==========================================

st.subheader("About the Project")

st.write(
    "This application uses a trained Machine Learning model "
    "to classify website characteristics as phishing or legitimate."
)

st.info(
    "Enter the 30 website feature values below. "
    "The model requires the same features used during training."
)

st.warning(
    "This version requires feature values as input. "
    "It does not automatically extract features from a URL. "
    "The default values are training-data medians, not measurements "
    "of the website being checked."
)


# ==========================================
# STEP 4: WEBSITE FEATURE INPUT FORM
# ==========================================

st.subheader("Enter Website Features")

with st.form("phishing_detection_form"):

    input_values = {}

    columns = st.columns(3)

    for index, feature in enumerate(feature_columns):

        # Use median as a starting value
        default_value = float(X_train[feature].median())

        # Get training minimum and maximum
        min_value = float(X_train[feature].min())
        max_value = float(X_train[feature].max())

        # Select suitable step size
        if (
            pd.api.types.is_integer_dtype(X_train[feature])
            or X_train[feature].nunique() <= 10
        ):
            step = 1.0
        else:
            step = 0.1

        with columns[index % 3]:

            input_values[feature] = st.number_input(
                label=feature,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=step,
                key=f"feature_{index}"
            )

    submit = st.form_submit_button(
        "🔍 Detect Phishing Website",
        use_container_width=True
    )


# ==========================================
# STEP 5: PREDICTION
# ==========================================

if submit:

    new_data = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    try:

        prediction = model.predict(new_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == -1:

            st.error(
                "⚠️ Phishing Website Detected!"
            )

            st.write(
                "The entered website characteristics "
                "are classified as phishing by the model."
            )

        elif prediction == 1:

            st.success(
                "✅ Legitimate Website"
            )

            st.write(
                "The entered website characteristics "
                "are classified as legitimate by the model."
            )

        else:

            st.warning("Unknown prediction result.")

        # Display phishing probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(new_data)

            class_index = list(model.classes_).index(-1)

            phishing_probability = (
                probabilities[0][class_index] * 100
            )

            st.metric(
                label="Phishing Probability",
                value=f"{phishing_probability:.2f}%"
            )

            st.progress(
                float(phishing_probability / 100)
            )

        st.caption(
            "This is a machine learning prediction, "
            "not a guarantee of website safety."
        )

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# ==========================================
# STEP 6: FOOTER
# ==========================================

st.divider()

st.caption(
    "MCA Final Year PBL Project | "
    "Phishing Website Detection Using Machine Learning"
)
