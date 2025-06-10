from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os

# Set Matplotlib to non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Load Pre-trained Model
model = load_model("model.keras")

# Create downloads directory if not exists
os.makedirs("downloads", exist_ok=True)

# User credentials
VALID_USERNAME = "renie"
VALID_PASSWORD = "050106"

# Helper Function to Convert Plot to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    plt.close(fig)  # Close the figure after saving to buffer to free memory
    return f"data:image/png;base64,{data}"

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    # Fetch stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")

    # Prepare test data (last 10% for testing)
    splitting_len = int(len(stock_data) * 0.9)
    test_subset = stock_data[['Close']][splitting_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(test_subset)

    # Check if there's enough data for prediction (at least 100 samples)
    if len(scaled_data) <= 100:
        return render_template("result.html", error="Not enough data to make predictions. Try a different stock or longer timeframe.")

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Model prediction
    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare DataFrame for plotting
    plotting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.flatten(),
        'Predicted Test Data': inv_predictions.flatten()
    }, index=test_subset.index[100:])

    # Plot 1: Closing Prices
    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(stock_data['Close'], 'b', label='Close Price')
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    original_plot = plot_to_html(fig1)

    # Plot 2: Original vs Predicted
    fig2 = plt.figure(figsize=(15, 6))
    plt.plot(plotting_data['Original Test Data'], label="Original Test Data")
    plt.plot(plotting_data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.legend()
    plt.title("Original vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    predicted_plot = plot_to_html(fig2)

    # Future Predictions
    last_100 = stock_data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100)

    future_predictions = []
    last_100_scaled = last_100_scaled.reshape(1, -1, 1)
    for _ in range(no_of_days):
        next_day = model.predict(last_100_scaled)
        future_predictions.append(scaler.inverse_transform(next_day)[0][0])
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)

    future_predictions = np.array(future_predictions).flatten()

    fig3 = plt.figure(figsize=(15, 6))
    plt.plot(range(1, no_of_days + 1), future_predictions, marker='o', label="Predicted Future Prices", color="purple")
    plt.title("Future Close Price Predictions")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.grid(alpha=0.3)
    plt.legend()
    future_plot = plot_to_html(fig3)

    return render_template(
        "result.html",
        stock=stock,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        enumerate=enumerate,
        future_predictions=future_predictions
    )

if __name__ == "__main__":
    app.run(debug=True)
