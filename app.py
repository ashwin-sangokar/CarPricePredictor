from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and dataset
try:
    model = pickle.load(open("car_price_predictor.pkl", "rb"))
    df = pd.read_csv("cleaned_car_data.csv")
except Exception as e:
    raise RuntimeError(f"Failed to load required files: {e}")

@app.route("/")
def index():
    company = sorted(df['company'].unique())
    car_models = {}

    for comp in df['company'].unique():
        car_models[comp] = sorted(df[df['company'] == comp]['name'].unique())

    years = sorted(df["year"].unique(), reverse=True)
    fuel = sorted(df['fuel_type'].unique())

    company.insert(0, "Select Company")

    return render_template(
        "index.html",
        companies=company,
        car_models=car_models,
        years=years,
        fuelType=fuel
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        company = request.form.get("company")
        carModel = request.form.get("carModel")
        year = int(request.form.get("year"))
        fuelType = request.form.get("fuelType")
        kmsDriven = int(request.form.get("odometer"))

        if not company or company == "Select Company":
            return "Please select a valid company"

        if not carModel:
            return "Please select car model"

        if kmsDriven <= 0:
            return "Invalid odometer reading"

        input_df = pd.DataFrame(
            [[carModel, company, year, kmsDriven, fuelType]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"]
        )

        prediction = model.predict(input_df)
        price = np.round(prediction[0], 2)

        if price < 0:
            return "Price prediction unavailable"

        return str(price)

    except ValueError:
        return "Invalid numeric input"
    except Exception as e:
        return f"Prediction failed: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
