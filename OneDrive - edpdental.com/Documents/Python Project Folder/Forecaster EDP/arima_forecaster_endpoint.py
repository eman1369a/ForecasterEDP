# -*- coding: utf-8 -*-

pip install flask statsmodels pmdarima sklearn

from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

app = Flask(__name__)

# Function to handle dynamic date selection and forecasting
@app.route('/forecast', methods=['POST'])
def forecast_sales():
    """
    API endpoint to forecast sales dynamically from forecast_start_date.
    Parameters:
    - sales_data: DataFrame containing the time series data in JSON format.
    - forecast_start_date: String date from which to start the forecast (inclusive).

    Returns:
    - JSON response with forecasted values and error metrics (MSE, MAE).
    """

    data = request.json
    sales_data = pd.DataFrame(data['sales_data']).set_index('Date')
    forecast_start_date = data['forecast_start_date']

    # Convert date column to DateTime and ensure business day frequency
    sales_data.index = pd.to_datetime(sales_data.index)
    sales_data = sales_data.asfreq('B')

    # Fill NaN values in 'Total' with zero
    sales_data['Total'].fillna(0, inplace=True)

    # Prepare training and testing data
    train_data = sales_data[:forecast_start_date]
    test_data = sales_data[forecast_start_date:]

    # Define and fit the SARIMAX model
    model = SARIMAX(train_data['Total'],
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 2, 5))
    results = model.fit()

    # Forecast from the forecast_start_date to the end of the data
    forecast_bdays = results.get_forecast(steps=len(test_data))
    forecast_mean_bdays = forecast_bdays.predicted_mean
    forecast_ci_bdays = forecast_bdays.conf_int()

    # Align the forecasted data with the test data
    forecast_mean_bdays.index = test_data.index
    forecast_ci_bdays.index = test_data.index

    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse_bdays = mean_squared_error(test_data['Total'], forecast_mean_bdays)
    mae_bdays = mean_absolute_error(test_data['Total'], forecast_mean_bdays)

    # Prepare response data
    response = {
        'forecast': forecast_mean_bdays.to_dict(),
        'confidence_interval': forecast_ci_bdays.to_dict(),
        'mse': mse_bdays,
        'mae': mae_bdays
    }

    return jsonify(response)

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)