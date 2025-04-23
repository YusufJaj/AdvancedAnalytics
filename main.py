from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from google.oauth2.service_account import Credentials
import pandas as pd
import gspread
import os
import json
import threading
import time
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load Google credentials from environment
google_creds = json.loads(os.environ['GOOGLE_CREDS_JSON'])
scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(google_creds, scopes=scopes)
gc = gspread.authorize(creds)

app = Flask(__name__)

# Global model storage
class WarehouseModels:
    def __init__(self):
        self.df = None
        self.sales_model = None
        self.shrink_model = None
        self.placement_model = None
        self.price_model = None
        self.label_encoder = None

    def retrain(self):
        logging.info("Retraining all models...")

        # Load data from Google Sheets
        worksheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1kyAE8sdc9Ekysc8H7mbrKtqffdlKDXjGsFr881Q1WIA").worksheet("Sheet1")
        data = worksheet.get_all_records()
        self.df = pd.DataFrame(data)

        df = self.df
        df['SalesVelocity'] = pd.to_numeric(df['SalesLast30Days'], errors='coerce').fillna(0) / 30
        df['ShrinkageRate'] = pd.to_numeric(df['ShrinkageLast30Days'], errors='coerce').fillna(0) / 30
        df['ForecastedSales'] = pd.to_numeric(df['ForecastedSales'], errors='coerce').fillna(0)
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0)

        # Forecasting Models
        X_sales = df[['SalesVelocity', 'StockLevel']]
        y_sales = pd.to_numeric(df['SalesLast30Days'], errors='coerce').fillna(0) * 1.2
        self.sales_model = LinearRegression().fit(X_sales, y_sales)

        X_shrink = df[['StockLevel', 'SalesLast30Days']].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_shrink = pd.to_numeric(df['ShrinkageLast30Days'], errors='coerce').fillna(0)
        self.shrink_model = LinearRegression().fit(X_shrink, y_shrink)

        # Shelf Optimization Model
        df['PlacementScore'] = df['ForecastedSales'] * df['UnitPrice']
        score_nonzero = df[df['PlacementScore'] > 0]
        df.loc[score_nonzero.index, 'PlacementPriority'] = pd.qcut(score_nonzero['PlacementScore'], q=3, labels=["Low", "Medium", "High"])

        self.label_encoder = LabelEncoder()
        df['CategoryEncoded'] = self.label_encoder.fit_transform(df['Category'].astype(str))
        X_place = df[['ForecastedSales', 'ShrinkageRate', 'UnitPrice', 'CategoryEncoded']].fillna(0)
        y_place = df['PlacementPriority'].astype(str)
        self.placement_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.placement_model.fit(X_place, y_place)

        # Auto-Pricing Model (Predict optimal price)
        X_price = df[['ForecastedSales', 'ShrinkageRate']].fillna(0)
        y_price = df['UnitPrice'].fillna(0)
        self.price_model = LinearRegression().fit(X_price, y_price)

        logging.info("Model retraining complete.")

models = WarehouseModels()

# Schedule daily retraining
def schedule_daily_retrain():
    def run():
        while True:
            now = datetime.now()
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            time.sleep((next_run - now).total_seconds())
            models.retrain()

    threading.Thread(target=run, daemon=True).start()

# Initial training
models.retrain()
schedule_daily_retrain()

# --- ROUTES ---

@app.route('/')
def home():
    return "Warehouse AI API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    stock = data['StockLevel']
    sales = data['SalesLast30Days']
    velocity = sales / 30
    forecast_sales = models.sales_model.predict([[velocity, stock]])[0]
    forecast_shrink = models.shrink_model.predict([[stock, sales]])[0]
    return jsonify({
        "ForecastedSales": round(forecast_sales),
        "ForecastedShrinkage": round(forecast_shrink, 2)
    })

@app.route('/reorder', methods=['POST'])
def reorder():
    data = request.json
    stock = data['StockLevel']
    reorder_point = data['ReorderPoint']
    forecast = data['ForecastedSales']
    reorder_qty = max(forecast - stock, reorder_point - stock)
    reorder_qty = max(0, int(reorder_qty))
    urgency = round((forecast - stock) / forecast * 10, 1) if forecast > 0 else 0
    urgency = min(10, max(0, urgency))
    recommended_date = "ASAP" if urgency >= 8 else "Monitor"
    return jsonify({
        "ReorderQuantity": reorder_qty,
        "ReorderUrgency": urgency,
        "RecommendedOrderDate": recommended_date
    })

@app.route('/shrinkage', methods=['POST'])
def shrinkage():
    data = request.json
    stock = data['StockLevel']
    sales = data['SalesLast30Days']
    shrinkage_last = data['ShrinkageLast30Days']
    predicted = round((shrinkage_last / max(sales, 1)) * sales * 1.1, 2)
    risk = "High" if predicted > 10 else "Medium" if predicted > 4 else "Low"
    flag = risk in ["High", "Medium"]
    return jsonify({
        "PredictedShrinkageNextMonth": predicted,
        "ShrinkageRiskLevel": risk,
        "FlagForInvestigation": flag
    })

@app.route('/placement', methods=['POST'])
def placement():
    try:
        data = request.json
        forecast = data['ForecastedSales']
        shrinkage_rate = data['ShrinkageRate']
        unit_price = data['UnitPrice']
        category = data.get('Category', 'Staples')
        if category not in models.label_encoder.classes_:
            category = models.label_encoder.classes_[0]
        category_encoded = models.label_encoder.transform([category])[0]
        prediction = models.placement_model.predict([[forecast, shrinkage_rate, unit_price, category_encoded]])[0]
        confidence = models.placement_model.predict_proba([[forecast, shrinkage_rate, unit_price, category_encoded]]).max()
        shelf_map = {"High": "Front Aisle", "Medium": "Mid Aisle", "Low": "Back Storage"}
        return jsonify({
            "SuggestedShelfLocation": shelf_map[prediction],
            "PlacementPriority": prediction,
            "HeatmapWeight": round(confidence * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profitability', methods=['POST'])
def profitability():
    try:
        data = request.json
        forecasted_sales = float(data['ForecastedSales'])
        unit_price = float(data['UnitPrice'])
        reorder_qty = float(data.get('ReorderQuantity', 1)) or 1
        revenue = forecasted_sales * unit_price
        score = 10 if revenue > 10000 else 8 if revenue > 5000 else 6 if revenue > 1000 else 4 if revenue > 500 else 2
        roi = revenue / reorder_qty
        return jsonify({
            "ForecastedRevenue": round(revenue, 2),
            "ProfitabilityScore": score,
            "RestockROI": round(roi, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/autoprice', methods=['POST'])
def autoprice():
    try:
        data = request.json
        forecasted_sales = float(data['ForecastedSales'])
        shrinkage_rate = float(data['ShrinkageRate'])
        input_features = [[forecasted_sales, shrinkage_rate]]
        suggested_price = models.price_model.predict(input_features)[0]
        return jsonify({
            "SuggestedUnitPrice": round(suggested_price, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app (for Render or gunicorn)
if __name__ == '__main__':
    app.run(debug=True)
