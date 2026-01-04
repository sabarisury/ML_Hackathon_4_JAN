import streamlit as st
import pandas as pd
import pickle 

# Load model + threshold
with open("best_xgb_model_with_threshold.pkl", "rb") as f:
    data = pickle.load(f)
    model = data['model']
    threshold = data['threshold']  
    
# Streamlit page config
st.set_page_config(page_title="HR Promotion Predictor", layout="centered")
st.title("🏢 HR Promotion Prediction System")
st.write("Enter employee details to predict promotion status")

# Input fields
department = st.selectbox("Department", [
    "Sales & Marketing", "Operations", "Technology",
    "Analytics", "Finance", "HR", "Procurement", "R&D", "Legal"
])

region = st.text_input("Region")
education = st.selectbox("Education", ["Below Secondary", "Bachelor's", "Master's & above"])
gender = st.selectbox("Gender", ["m", "f"])
recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing", "referred", "other"])

no_of_trainings = st.number_input("No. of Trainings", 0, 10)
age = st.number_input("Age", 18, 60)
previous_year_rating = st.slider("Previous Year Rating", 1, 5)
length_of_service = st.number_input("Length of Service (Years)", 0, 40)
KPIs_met = st.selectbox("KPIs Met > 80%", [0, 1])
awards_won = st.selectbox("Awards Won", [0, 1])
avg_training_score = st.slider("Avg Training Score", 0, 100)

if st.button("Predict Promotion"):
    # Create input dataframe matching the training columns
    input_data = pd.DataFrame([{
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": KPIs_met,
        "awards_won?": awards_won,
        "avg_training_score": avg_training_score
    }])

    # Predict probability and apply threshold
    prob = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prob >= threshold)

    # Display result
    st.subheader("Prediction Result")
    st.write(f"Promotion Probability: **{prob:.2f}**")

    if prediction == 1:
        st.success("🎉 Employee is likely to be PROMOTED!!!!!!!!!!")
    else:
        st.warning("❌ Employee is NOT likely to be promoted!!!!!!!")