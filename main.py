from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from google.oauth2.service_account import Credentials
import pandas as pd
import gspread
import os
import json

# Load Google Sheets credentials from environment variable
google_creds = json.loads(os.environ['GOOGLE_CREDS_JSON'])
scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(google_creds, scopes=scopes)

# Authorize and fetch data from Google Sheets
gc = gspread.authorize(creds)
spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1kyAE8sdc9Ekysc8H7mbrKtqffdlKDXjGsFr881Q1WIA")
worksheet = spreadsheet.worksheet("Sheet1")
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Feature engineering
df['SalesVelocity'] = df['SalesLast30Days'] / 30
df['ShrinkageRate'] = df['ShrinkageLast30Days'] / 30

# SALES & SHRINKAGE PREDICTION
X_sales = df[['SalesVelocity', 'StockLevel']]
y_sales = df['SalesLast30Days'] * 1.2
sales_model = LinearRegression().fit(X_sales, y_sales)

X_shrink = df[['StockLevel', 'SalesLast30Days']]
y_shrink = df['ShrinkageLast30Days']
shrink_model = LinearRegression().fit(X_shrink, y_shrink)

# SHELF PLACEMENT MODEL
df['PlacementScore'] = df['ForecastedSales'].fillna(0) * df['UnitPrice'].fillna(0)
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

# Flask API
app = Flask(__name__)

@app.route('/')
def home():
    return "Warehouse AI API is running!"

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
    sales = data['SalesLast30Days']
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
            category = 'Other' if 'Other' in label_encoder.classes_ else label_encoder.classes_[0]
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
        print("Placement route error:", str(e))
        return jsonify({"error": str(e)}), 500