
from flask import Flask, render_template, flash, request, session, send_file
import pickle
import numpy as np
import mysql.connector
import sys

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Home")
def Home():
    return render_template('index.html')


@app.route("/Predict")
def Predict():
    return render_template('Predict.html')


@app.route("/pred", methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        BloodType = request.form['BloodType']
        MedicalCondition = request.form['MedicalCondition']
        AdmissionType = request.form['AdmissionType']
        Medication = request.form['Medication']


        t1 = int(age)
        t2 = int(gender)
        t3 = int(BloodType)
        t4 = int(MedicalCondition)
        t5 = int(AdmissionType)
        t6 = float(Medication)



        filename2 = "./Dataset/rfc-model.pkl"
        classifier2 = pickle.load(open(filename2, 'rb'))

        data = np.array([[t1, t2, t3, t4, t5, t6]])
        my_prediction = classifier2.predict(data)
        print(my_prediction[0])

        if my_prediction == 1:
            Answer = "Inconclusive"

        elif  my_prediction == 2:
            Answer = "Normal"
        elif  my_prediction == 0:
            Answer = "Abnormal"

        return render_template('Result.html', data=Answer)


@app.route("/ViewDoctor", methods=['GET', 'POST'])
def ViewDoctor():
    if request.method == 'POST':
        return render_template('Predict.html')




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
