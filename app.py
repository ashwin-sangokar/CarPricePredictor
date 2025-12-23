from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle 


app = Flask(__name__)

model = pickle.load(open("car_price_predictor.pkl","rb")) 
df = pd.read_csv("cleaned_car_data.csv")

@app.route('/')
def index() :

    company = sorted(df['company'].unique())
    car_models = {}
    for comp in df['company'].unique():
        car_models[comp] = sorted(df[df['company'] == comp]['name'].unique())
    year = sorted(df["year"].unique(), reverse=True)
    fuel = sorted(df['fuel_type'].unique())
    company.insert(0, "Select Company")
    return render_template("index.html", companies=company, car_models=car_models, years=year, fuelType=fuel)

@app.route('/predict', methods=['POST'])
def predict() :
    company = request.form.get("company")
    carModel = request.form.get("carModel")
    year = int(request.form.get("year"))
    fuelType = request.form.get("fuelType")
    kmsDriven = int(request.form.get("odometer"))

    prediction = model.predict(pd.DataFrame([[carModel, company, year, kmsDriven, fuelType]], columns=["name", "company", "year", "kms_driven", "fuel_type"]))
    output = np.round(prediction[0], 2)
    return str(output)
    




if __name__ == "__main__" :
    app.run(debug=True)