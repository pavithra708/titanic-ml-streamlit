import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('titanic_model.pkl')

# App title and intro
st.title('🚢 Titanic Survival Predictor')
st.markdown("""
This mini ML app predicts whether a Titanic passenger would have survived  
based on their class, age, sex, fare, and family aboard.
""")

st.markdown("---")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox('🎟 Passenger Class', [1, 2, 3], index=0)
    sex = st.selectbox('👤 Sex', ['male', 'female'], index=1)
    age = st.slider('🎂 Age', 0, 80, 20)
    fare = st.number_input('💰 Fare', 0.0, 300.0, 100.0)

with col2:
    sibsp = st.number_input('🧑‍🤝‍🧑 Number of siblings/spouses aboard', 0, 8, 1)
    parch = st.number_input('👨‍👩‍👧‍👦 Number of parents/children aboard', 0, 6, 1)
    embarked = st.selectbox('🛳 Port of Embarkation', ['S', 'C', 'Q'], index=0)

# Encode inputs
sex_encoded = 0 if sex == 'male' else 1
embarked_encoded = {'S': 0, 'C': 1, 'Q': 2}[embarked]

features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict button
if st.button('🔍 Predict Survival'):
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Probability of survival
    
    if prediction[0] == 1:
        st.success(f"✅ Survived (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Did not survive (Confidence: {1 - probability:.2%})")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn")
