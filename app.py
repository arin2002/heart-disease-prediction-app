import streamlit as st
import numpy as np
import joblib

# import model
model = joblib.load('./heart_disease_model.pkl')

# Define dictionaries for labels
sex_dict = {'Male': 1, 'Female': 0}
cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1,
           'Non-Anginal Pain': 2, 'Asymptomatic': 3}
fasting_bs_dict = {'False': 0, 'True': 1}
restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1,
                'Left Ventricular Hypertrophy': 2}
slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thal_dict = {'Normal': 0, 'Fixed Defect': 1, 'Reversable Defect': 2}
exercise_angina_dict = {'True':1,'False':0}

# Define function to convert labels to index
def label_to_index(label, label_dict):
    return label_dict[label]

# Define function to convert index to labels
def index_to_label(index, label_dict):
    for label, i in label_dict.items():
        if i == index:
            return label


# Set page title and favicon
st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon=":heartbeat:")

# Set page header
st.write("""
    <h1 style="color:#D42027;text-align:center;">Heart Disease Prediction</h1>
    <p style="color:#585858;text-align:center;">Enter patient details and click 'Predict' to check for heart disease.</p>
    <hr style="height:5px;border:none;color:#D42027;background-color:#D42027;">
    """, unsafe_allow_html=True)

# Get user inputs
st.sidebar.header('User Input Features')
age = st.sidebar.slider('Age', 18, 100, 50)
sex = st.sidebar.selectbox('Sex', list(sex_dict.keys()))
cp = st.sidebar.selectbox('Chest Pain Type', list(cp_dict.keys()))
resting_bp = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.sidebar.slider('Serum Cholestoral in mg/dl', 100, 600, 200)
fasting_bs = st.sidebar.selectbox(
    'Fasting Blood Sugar > 120 mg/dl', list(fasting_bs_dict.keys()))
restecg = st.sidebar.selectbox('Resting ECG', list(restecg_dict.keys()))
max_hr = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 120)
exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ['False', 'True'])
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 10.0, 2.0)
slope = st.sidebar.selectbox('Slope', list(slope_dict.keys()))
major_vessels = st.sidebar.slider('Number of Major Vessels', 0, 3, 0)
thal = st.sidebar.selectbox('Thal', list(thal_dict.keys()))


# Add predict button
if st.button('Predict'):
    # Convert labels to index
    sex = label_to_index(sex, sex_dict)
    cp = label_to_index(cp, cp_dict)
    fasting_bs = label_to_index(fasting_bs, fasting_bs_dict)
    restecg = label_to_index(restecg, restecg_dict)
    slope = label_to_index(slope, slope_dict)
    thal = label_to_index(thal, thal_dict)
    exercise_angina = label_to_index(exercise_angina,exercise_angina_dict)

    # Load trained model and predict
    st.write(age, sex, cp, resting_bp, cholesterol, fasting_bs, restecg,
             max_hr, exercise_angina, oldpeak, slope, major_vessels, thal)
    input_data = (age, sex, cp, resting_bp, cholesterol, fasting_bs, restecg,
                  max_hr, exercise_angina, int(oldpeak), slope, major_vessels, thal)
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)

    # Convert index back to labels for display
    sex = index_to_label(sex, sex_dict)
    cp = index_to_label(cp, cp_dict)
    fasting_bs = index_to_label(fasting_bs, fasting_bs_dict)
    restecg = index_to_label(restecg, restecg_dict)
    slope = index_to_label(slope, slope_dict)
    thal = index_to_label(thal, thal_dict)

    # Define colors
    green_color = '#1abc9c'
    red_color = '#e74c3c'
    blue_color = '#3498db'

    # Display prediction result with colors
    if prediction[0] == 1:
        st.write('Prediction: ',
                 f'<span style="color:{red_color}; font-size:18px;">Heart Disease</span>', unsafe_allow_html=True)
    else:
        st.write('Prediction: ',
                 f'<span style="color:{green_color}; font-size:18px;">No Heart Disease</span>', unsafe_allow_html=True)

    # Display user inputs with labels and colors
    st.write('User Inputs:')
    st.write(
        f'Age: <span style="color:{blue_color}">{age}</span>', unsafe_allow_html=True)
    st.write(
        f'Sex: <span style="color:{blue_color}">{sex}</span>', unsafe_allow_html=True)
    st.write(
        f'Chest Pain Type: <span style="color:{blue_color}">{cp}</span>', unsafe_allow_html=True)
    st.write(
        f'Resting Blood Pressure: <span style="color:{blue_color}">{resting_bp}</span>', unsafe_allow_html=True)
    st.write(
        f'Serum Cholestoral in mg/dl: <span style="color:{blue_color}">{cholesterol}</span>', unsafe_allow_html=True)
    st.write(
        f'Fasting Blood Sugar > 120 mg/dl: <span style="color:{blue_color}">{fasting_bs}</span>', unsafe_allow_html=True)
    st.write(
        f'Resting ECG: <span style="color:{blue_color}">{restecg}</span>', unsafe_allow_html=True)
    st.write(
        f'Maximum Heart Rate Achieved: <span style="color:{blue_color}">{max_hr}', unsafe_allow_html=True)
    st.write(
        f'Exercise Induced Angina: <span style="color:{blue_color}">{exercise_angina}</span>', unsafe_allow_html=True)
    st.write(
        f'Oldpeak: <span style="color:{blue_color}">{oldpeak}', unsafe_allow_html=True)
    st.write(
        f'Slope: <span style="color:{blue_color}">{slope}</span>', unsafe_allow_html=True)
    st.write(
        f'Number of Major Vessels: <span style="color:{blue_color}">{major_vessels}',unsafe_allow_html=True)
    st.write(
        f'Thal: <span style="color:{blue_color}">{thal}</span>', unsafe_allow_html=True)
