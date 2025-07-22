import streamlit as st
import joblib
import pandas as pd

# Load model and encoders (make sure to place these files in the same folder)
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

def go_to_result():
    st.session_state.page = 'result'

def go_back_to_form():
    st.session_state.page = 'form'

def predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list):
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

    for col in model.feature_names_in_:
        if col not in final_input.columns:
            final_input[col] = 0
    final_input = final_input[model.feature_names_in_]

    predicted = model.predict(final_input)[0]
    return round(predicted, 2)

# Form Page
if st.session_state.page == 'form':
    st.title("💼 Salary Prediction Tool")

    job_title = st.selectbox("Job Title", label_encoders['job_title'].classes_)
    years_of_experience = st.number_input("Years of Experience", 0, 50, 2)
    location = st.selectbox("Location", label_encoders['location'].classes_)
    education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
    company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)
    skills_list = st.multiselect("Skills", mlb.classes_)

    if st.button("Predict My Salary"):
        salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list)
        st.session_state.predicted_salary = salary
        st.session_state.user_inputs = {
            'Position': job_title,
            'Experience': years_of_experience,
            'Location': location,
            'Education': education_level
        }
        go_to_result()

# Result Page
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=go_back_to_form)

    st.markdown(f"""
    <div style="background: linear-gradient(to right, #00c6ff, #0072ff); padding: 30px; border-radius: 10px; color: white; text-align: center;">
        <h2>💲 Salary Prediction</h2>
        <h1>${st.session_state.predicted_salary:,.0f}</h1>
        <p>Annual Salary Estimate</p>
        <p><b>${st.session_state.predicted_salary * 0.85:,.0f}</b> Low Range — <b>${st.session_state.predicted_salary * 1.15:,.0f}</b> High Range</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; gap: 30px; margin-top: 20px;">
        <div style="flex: 1; background: #f9f9f9; padding: 20px; border-radius: 10px;">
            <h4>🎯 Your Profile</h4>
    """, unsafe_allow_html=True)

    for key, value in st.session_state.user_inputs.items():
        st.markdown(f"**{key}:** {value}")

    st.markdown("""
        </div>
        <div style="flex: 1; background: #f9f9f9; padding: 20px; border-radius: 10px;">
            <h4>📊 Market Insights</h4>
            <p>**Industry Average:** $55,200</p>
            <p>**Top 10% Earners:** $87,000</p>
            <p>**Growth Potential:** <span style='color:green;'>High</span></p>
            <p>**Demand Level:** <span style='color:blue;'>Very High</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
