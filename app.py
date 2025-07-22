import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

st.set_page_config(page_title="Salary Prediction Tool", layout="centered")

st.title("💼 Salary Prediction Tool")
st.write("Enter your details to get an accurate salary prediction")

# Input fields
job_title = st.selectbox("Job Title", label_encoders['job_title'].classes_)
years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
location = st.selectbox("Location", label_encoders['location'].classes_)
education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)

# Skills Multiselect
skills_list = st.multiselect(
    "Skills & Technologies",
    mlb.classes_
)

# Predict button
if st.button("Predict My Salary"):
    # Prepare input
    input_data = {
        'job_title': label_encoders['job_title'].transform([job_title])[0],
        'years_of_experience': years_of_experience,
        'location': label_encoders['location'].transform([location])[0],
        'education_level': label_encoders['education_level'].transform([education_level])[0],
        'company_size': label_encoders['company_size'].transform([company_size])[0]
    }
    skills_encoded = mlb.transform([skills_list])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)
    input_df = pd.DataFrame([input_data])
    final_input = pd.concat([input_df, skills_df], axis=1)

    # Align columns
    for col in model.feature_names_in_:
        if col not in final_input.columns:
            final_input[col] = 0
    final_input = final_input[model.feature_names_in_]

    # Prediction
    predicted_salary = model.predict(final_input)[0]
    st.success(f"💰 Estimated Salary: **${round(predicted_salary, 2)}**")
