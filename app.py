from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Manual encoding dictionaries
business_travel_encoding = {0: 'Travel_Rarely', 1: 'Travel_Frequently', 2: 'Non-Travel'}
department_encoding = {0: 'Research & Development', 1: 'Sales', 2: 'Human Resources'}
education_field_encoding = {0: 'Life Sciences', 1: 'Other', 2: 'Medical', 3: 'Marketing', 4: 'Technical Degree', 5: 'Human Resources'}
overtime_encoding = {0: 'No', 1: 'Yes'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'BusinessTravel': int(request.form['BusinessTravel']),
            'Department': int(request.form['Department']),
            'DistanceFromHome': float(request.form['DistanceFromHome']),
            'EducationField': int(request.form['EducationField']),
            'JobLevel': float(request.form['JobLevel']),
            'JobSatisfaction': float(request.form['JobSatisfaction']),
            'OverTime': int(request.form['OverTime']),
            'StockOptionLevel': float(request.form['StockOptionLevel']),
            'TotalWorkingYears': float(request.form['TotalWorkingYears']),
            'YearsInCurrentRole': float(request.form['YearsInCurrentRole'])
        }

        # Create a DataFrame
        new_data = pd.DataFrame([data])

        # Scale the data
        new_data_scaled = scaler.transform(new_data)

        # Predict
        new_prediction = ensemble_model.predict(new_data_scaled)
        attrition_prediction = 'Yes' if new_prediction[0] == 1 else 'No'

        return render_template('result.html', prediction=attrition_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)