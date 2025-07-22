import streamlit as st
import joblib
import pandas as pd

# ----------- Load model and encoders -----------
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# ----------- Session State Setup -----------
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

# ----------- Custom CSS -----------
st.markdown("""
<style>
body { background: #101217; }
.header-main {
    max-width: 600px;
    margin: 42px auto 30px auto;
    border-radius: 22px;
    text-align: center;
    position: relative;
}
.header-box {
    background: linear-gradient(90deg, #005bea 0%, #00c6fb 100%);
    border-radius: 22px;
    padding: 32px 0 25px 0;
    box-shadow: 0 8px 40px 0 rgba(0,40,120,0.13);
    margin-bottom: 0;
    border: 1.5px solid #ddefff;
    position: relative;
}
.header-box .page-title {
    position: absolute;
    left: 32px;
    top: 20px;
    color: #e6f4ff;
    font-size: 1em;
    font-weight: 600;
    letter-spacing: 0.01em;
    opacity: 0.85;
    margin: 0;
    padding: 0;
    text-align: left;
}
.header-title {
    color: #fff;
    font-size: 2.2em;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 12px;
    margin-top: 0;
    text-shadow: 0 2px 18px rgba(0,170,255,0.12);
}
.header-desc {
    color: #e6f4ff;
    font-size: 1.08em;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 0;
    letter-spacing: 0.003em;
}
.result-cards-row {
    display: flex;
    gap: 32px;
    justify-content: center;
    margin-top: 18px;
}
.card-box {
    background: #fff;
    border-radius: 22px;
    box-shadow: 0 4px 24px rgba(0,80,255,0.10);
    padding: 28px 26px 18px 26px;
    color: #222;
    min-width: 260px;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 18px;
}
@media (max-width: 900px) {
    .result-cards-row { flex-direction: column; gap: 12px; align-items:center;}
    .card-box {width:90vw; min-width:0;}
}
@media (max-width: 700px) {
    .header-main { max-width: 96vw; }
    .header-box { padding: 16px 0 12px 0; }
    .header-title { font-size: 1.25em; }
    .header-desc { font-size: 0.99em; }
    .header-box .page-title { left: 12px; top: 8px; font-size: 0.93em; }
    .card-box {padding:18px 10px;}
}
.salary-prediction-card {
    background: #fff;
    border-radius: 22px;
    box-shadow: 0 4px 24px rgba(0,80,255,0.10);
    padding: 32px 24px 28px 24px;
    margin: 34px auto 0 auto;
    color: #222;
    max-width: 480px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
# ---------------------- FORM PAGE ---------------------- #
if st.session_state.page == 'form':
    st.markdown("""
    <div class="header-main">
      <div class="header-box">
        <div class="page-title">Salary Predictor</div>
        <div class="header-title">Salary Prediction Tool</div>
        <div class="header-desc">
            Get your personalized salary estimate instantly.<br>
            Enter your details below to see your market value!
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("salary_form"):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            job_title = st.selectbox("🏢 Job Title", label_encoders['job_title'].classes_)
            years_of_experience = st.number_input("⏳ Years of Experience", 0, 50, 2, help="Enter total years of professional experience.")
            location = st.selectbox("📍 Location", label_encoders['location'].classes_)
            education_level = st.selectbox("🎓 Education Level", label_encoders['education_level'].classes_)
            company_size = st.selectbox("🏢 Company Size", label_encoders['company_size'].classes_)
            skills_list = st.multiselect(
                "💡 Select Your Skills",
                mlb.classes_,
                help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔎 Predict My Salary")
        if submitted:
            salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list)
            st.session_state.predicted_salary = salary
            st.session_state.user_inputs = {
                'Position': job_title,
                'Experience': years_of_experience,
                'Location': location,
                'Education': education_level
            }
            go_to_result()

# ---------------------- RESULT PAGE ---------------------- #
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=go_back_to_form)

    # Cards row for profile and market insights
    st.markdown("""
    <div class="result-cards-row">
      <div class="card-box">
        <div style="color:#1abcfe; font-weight:700; margin-bottom:12px; font-size:1.08em;">
            <span style="font-size:1.1em;">🎯</span>
            <span style="color:#1abcfe; font-weight:700;">Your Profile</span>
        </div>
        <div style="font-size:1.07em; margin-left:3px;">
            <b>Position:</b> {position}<br>
            <b>Experience:</b> {experience}<br>
            <b>Location:</b> {location}<br>
            <b>Education:</b> {education}
        </div>
      </div>
      <div class="card-box">
        <div style="color:#1abcfe; font-weight:700; margin-bottom:12px; font-size:1.08em;">
            <span style="font-size:1.1em;">📊</span>
            <span style="color:#1abcfe; font-weight:700;">Market Insights</span>
        </div>
        <div style="font-size:1.07em; margin-left:3px;">
            <b>Industry Average:</b> $55,200<br>
            <b>Top 10% Earners:</b> $87,000<br>
            <b>Growth Potential:</b> <span style='color:#1aa260;font-weight:700;'>High</span><br>
            <b>Demand Level:</b> <span style='color:#1a56e0;font-weight:700;'>Very High</span>
        </div>
      </div>
    </div>
    """.format(
        position=st.session_state.user_inputs.get('Position', ''),
        experience=st.session_state.user_inputs.get('Experience', ''),
        location=st.session_state.user_inputs.get('Location', ''),
        education=st.session_state.user_inputs.get('Education', '')
    ), unsafe_allow_html=True)

    # Salary Prediction in a box below
    st.markdown(f"""
        <div class="salary-prediction-card">
            <h2 style="color:#16C60C; margin-bottom:18px;">
                <span style="font-size:1.3em;">💲</span> Salary Prediction
            </h2>
            <h1 style="margin-bottom: 0.2em; font-size:2.4em; color:#222;">
                ${st.session_state.predicted_salary:,.0f}
            </h1>
            <div style="font-size:1.17em; color:#222; margin-bottom:10px;">
                Estimated Annual Salary
            </div>
            <div style="font-size:1.08em;">
                <b>${st.session_state.predicted_salary * 0.85:,.0f}</b> Low — 
                <b>${st.session_state.predicted_salary * 1.15:,.0f}</b> High
            </div>
        </div>
    """, unsafe_allow_html=True)
