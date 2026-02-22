import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================= Load Model & Scaler =================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Employee Retention Prediction", layout="centered")

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee will **Stay** or **Leave** the company")

st.divider()

# ================= User Input =================
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Part time course", "Full time course"])
    education_level = st.selectbox("Education Level", ["Graduate", "Masters", "High School", "Phd", "Primary School"])
    major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "Other", "No Major"])
    experience = st.selectbox("Experience", ["<1"] + [str(i) for i in range(1, 21)] + [">20"])
    company_size = st.selectbox("Company Size", ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
    company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"])
    last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4"])
    training_hours = st.slider("Training Hours", 0, 350, 40)

    data = {
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours
    }

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# ================= Data Preprocessing =================
df = input_df.copy()

# Label Encoding (same logic as training)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Dummy Encoding (same as training)
df = pd.get_dummies(df, drop_first=True)

# Align columns with training model
model_features = model.feature_names_in_
df = df.reindex(columns=model_features, fill_value=0)

# ================= Prediction =================
if st.button("🔍 Predict Employee Status"):
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("❌ Employee is likely to **LEAVE** the company")
    else:
        st.success("✅ Employee is likely to **STAY** in the company")

    st.write(f"📊 **Probability of Leaving:** `{prediction_proba:.2f}`")

st.divider()
st.caption("Model Used: Random Forest Classifier (Optimized)")






























