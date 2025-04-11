from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from google.oauth2.service_account import Credentials
import pandas as pd
import gspread
import os
import json

# Load Google Sheets credentials from Render environment variable
google_creds = json.loads(os.environ['GOOGLE_CREDS_JSON'])
scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(google_creds, scopes=scopes)

# Authorize gspread
gc = gspread.authorize(creds)

# Connect to the Google Sheet
spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1kyAE8sdc9Ekysc8H7mbrKtqffdlKDXjGsFr881Q1WIA")
worksheet = spreadsheet.worksheet("Sheet1")
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Feature engineering
df['SalesVelocity'] = pd.to_numeric(df['SalesLast30Days'], errors='coerce').fillna(0) / 30
df['ShrinkageRate'] = pd.to_numeric(df['ShrinkageLast30Days'], errors='coerce').fillna(0) / 30

# Sales and shrinkage models
X_sales = df[['SalesVelocity', 'StockLevel']]
y_sales = pd.to_numeric(df['SalesLast30Days'], errors='coerce').fillna(0) * 1.2
sales_model = LinearRegression().fit(X_sales, y_sales)

X_shrink = df[['StockLevel', 'SalesLast30Days']]
X_shrink = X_shrink.apply(pd.to_numeric, errors='coerce').fillna(0)
y_shrink = pd.to_numeric(df['ShrinkageLast30Days'], errors='coerce').fillna(0)
shrink_model = LinearRegression().fit(X_shrink, y_shrink)

# Convert key columns to numeric
df['ForecastedSales'] = pd.to_numeric(df['ForecastedSales'], errors='coerce').fillna(0)
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0)

# Shelf optimization model
df['PlacementScore'] = df['ForecastedSales'] * df['UnitPrice']
score_nonzero = df[df['PlacementScore'] > 0]
df.loc[score_nonzero.index, 'PlacementPriority'] = pd.qcut(
    score_nonzero['PlacementScore'],
    q=3,
    labels=["Low", "Medium", "High"]
)

label_encoder = LabelEncoder()
df['CategoryEncoded'] = label_encoder.fit_transform(df['Category'].astype(str))
X_place = df[['ForecastedSales', 'ShrinkageRate', 'UnitPrice', 'CategoryEncoded']].fillna(0)
y_place = df['PlacementPriority'].astype(str)
placement_model = RandomForestClassifier(n_estimators=100, random_state=42)
placement_model.fit(X_place, y_place)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return "Warehouse AI API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    stock = data['StockLevel']
    sales = data['SalesLast30Days']
    velocity = sales / 30
    forecast_sales = sales_model.predict([[velocity, stock]])[0]
    forecast_shrink = shrink_model.predict([[stock, sales]])[0]
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
    category = data.get('Category', 'General')
    predicted = round((shrinkage_last / max(sales, 1)) * sales * 1.1, 2)
    if predicted > 10:
        risk = "High"
    elif predicted > 4:
        risk = "Medium"
    else:
        risk = "Low"
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
        category = data.get('Category', 'General')
        if category not in label_encoder.classes_:
            category = 'Staples'
        category_encoded = label_encoder.transform([category])[0]
        input_features = [[forecast, shrinkage_rate, unit_price, category_encoded]]
        prediction = placement_model.predict(input_features)[0]
        confidence = placement_model.predict_proba(input_features).max()
        shelf_map = {
            "High": "Front Aisle",
            "Medium": "Mid Aisle",
            "Low": "Back Storage"
        }
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

        # Forecasted Revenue
        revenue = forecasted_sales * unit_price

        # Scaled score from 1 to 10
        if revenue > 10000:
            score = 10
        elif revenue > 5000:
            score = 8
        elif revenue > 1000:
            score = 6
        elif revenue > 500:
            score = 4
        else:
            score = 2

        # ROI = Revenue per reordered unit
        roi = revenue / reorder_qty

        return jsonify({
            "ForecastedRevenue": round(revenue, 2),
            "ProfitabilityScore": score,
            "RestockROI": round(roi, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
