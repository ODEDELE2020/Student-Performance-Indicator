from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    gender_new = float(request.form['gender_new'])
    race_ethnicity_new = float(request.form['race_ethnicity_new'])
    parental_level_of_education_new= float(request.form['parental_level_of_education_new'])
    lunch_new= float(request.form['lunch_new'])
    test_preparation_course_new = float(request.form['test_preparation_course_new'])
    reading_score = float(request.form['reading_score'])
    writing_score= float(request.form['writing_score'])
    

    X= np.array([[gender_new,race_ethnicity_new,parental_level_of_education_new,
                  lunch_new,test_preparation_course_new,reading_score,writing_score]])

    model_path=r'C:\Users\Pelux\Desktop\Jupyter Notebook\STUDENT PERFORMANCE\finalized_model.sav'
    model= joblib.load(model_path)

    y_pred = model.predict(X_test)

    return jsonify({'Prediction': float(y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)
