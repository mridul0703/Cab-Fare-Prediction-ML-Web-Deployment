from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from datetime import datetime
import numpy as np

app = Flask(__name__)

def load_train_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['pickup_datetime'])
    return df

def preprocess_data(df):
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.weekday
    X = df[['pickup_hour', 'pickup_day_of_week', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
    return X

def train_model(X, y, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'linear_svr':
        model = LinearSVR()
    elif model_type == 'bagging':
        model = BaggingRegressor()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_type == 'neural_network':
        model = MLPRegressor()
    elif model_type == 'hist_gradient_boosting':
        model = GradientBoostingRegressor()
    elif model_type == 'voting':
        model = VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor()), ('xgb', XGBRegressor())])
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError("Invalid model type")
    model.fit(X, y)
    return model

def preprocess_input(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    pickup_datetime = datetime.strptime(pickup_datetime, '%Y-%m-%dT%H:%M')
    pickup_hour = pickup_datetime.hour
    pickup_day_of_week = pickup_datetime.weekday()
    X = [[pickup_hour, pickup_day_of_week, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]]
    return X

def predict_fare_amount(X, model):
    fare_amount = model.predict(X)
    return fare_amount

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pickup_datetime = request.form['pickup_datetime']
        pickup_longitude = float(request.form['pickup_longitude'])
        pickup_latitude = float(request.form['pickup_latitude'])
        dropoff_longitude = float(request.form['dropoff_longitude'])
        dropoff_latitude = float(request.form['dropoff_latitude'])
        passenger_count = int(request.form['passenger_count'])
        model_type = request.form['model_type']
        train_file_path = 'train.csv'  # Update with your train data file path
        train_df = load_train_data(train_file_path)
        X_train = preprocess_data(train_df)
        y_train = train_df['fare_amount']
        model = train_model(X_train, y_train, model_type)
        X_input = preprocess_input(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count)
        fare_amount = predict_fare_amount(X_input, model)
        fare_amount = (fare_amount * passenger_count).tolist()
        return render_template('index.html', fare="$ - {:.2f}".format(fare_amount[0]))

if __name__ == '__main__':
    app.run(debug=True)
