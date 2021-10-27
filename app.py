import numpy as np
from flask import Flask, render_template, request,jsonify
import pickle

app = Flask(__name__)

rf = pickle.load(open('cardio.pkl','rb')) #pickle object


@app.route('/')
def home():
    return "hello"

@app.route('/predict', methods = ["POST"])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    chest_pain_type = request.form.get('chest_pain_type')
    resting_bp_s = request.form.get('resting_bp_s')
    cholesterol = request.form.get('cholesterol')
    fasting_blood_sugar = request.form.get('fasting_blood_sugar')
    resting_ecg = request.form.get('resting_ecg')
    max_heart_rate = request.form.get('max_heart_rate')
    exercise_angina = request.form.get('exercise_angina')
    oldpeak = request.form.get('oldpeak')
    ST_slope = request.form.get('ST_slope')

    input_array = np.array([[age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope]])
    prediction = rf.predict(input_array)[0]

    return jsonify({'prediction': str(prediction)})

    
if __name__ == '__main__':
    app.run(debug=True)
