
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for
import joblib

app = Flask(__name__)
model = joblib.load(r"C:\PROJECTS_FILES\Student_Analysis.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    #Get Input Values
    
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)

    #Validate Input Hours

    if input_features[0] < 0 or input_features[0] > 24:
        return render_template('index.html', 
                               prediction_text='please enter valid number'
                               )
    #Prediction
    #output = round(model.predict([features_value])[0],2)
    prediction = model.predict([features_value])
    output = round(float(prediction[0]),2)
    
    
    #input and predicted values sotred in df then save in csv file
    
    new_data = pd.DataFrame({
        'Study_Hours':[input_features[0]],
        'Predicted_Output':[output]
        })
    
    df = pd.concat([df, new_data], ignore_index =True)
    
    print(df)
    
    #Save CSV
    
    df.to_csv('smp_data_from_app.csv', index=False)
    
    #Return Output
    
    return render_template(
        'index.html',
        prediction_text =f'You Will get approximtely {output} marks'
        )
    # Run app
    
if __name__ == "__main__":
    print("Flask app started")
    app.run(host='127.0.0.1', port=5002, debug=True, use_reloader = False)
    
    