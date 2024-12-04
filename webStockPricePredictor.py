import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "AAPL")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

stock_data = yf.download(stock, start, end)
stock_data.columns = [col[0] for col in stock_data.columns]

model = load_model("models\\Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(stock_data)

splitting_len = int(len(stock_data) * 0.7)
x_test = pd.DataFrame(stock_data.Close[splitting_len:])

def plot_combined_graph(figsize, full_data, *moving_averages):
    """
    Plot the original close price along with one or more moving averages.
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data.Close, 'b', label='Original Close Price')
    
    # Add moving averages if provided
    for ma, label, color in moving_averages:
        plt.plot(ma, color, label=label)
    
    plt.legend(loc='best')
    plt.title('Original Close Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    return fig

# Calculate moving averages
stock_data['MA_for_100_days'] = stock_data.Close.rolling(100).mean()
stock_data['MA_for_200_days'] = stock_data.Close.rolling(200).mean()
stock_data['MA_for_250_days'] = stock_data.Close.rolling(250).mean()

# Plot all in a single graph
st.subheader('Original Close Price with Moving Averages')
st.pyplot(plot_combined_graph(
    (15, 6), 
    stock_data,
    (stock_data['MA_for_100_days'], 'MA for 100 Days', 'orange'),
    (stock_data['MA_for_200_days'], 'MA for 200 Days', 'green'),
    (stock_data['MA_for_250_days'], 'MA for 250 Days', 'red')
))

# RSI
def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given data series.
    :param data: pandas Series, the price data (e.g., Close prices).
    :param window: int, the lookback period for RSI calculation.
    :return: pandas Series, the RSI values.
    """
    delta = data.diff(1)  # Calculate daily price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Calculate average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Calculate average loss

    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

# Add RSI to stock data
stock_data['RSI'] = calculate_rsi(stock_data.Close)

# Plot RSI
def plot_rsi_graph(figsize, full_data):
    """
    Plot the RSI along with overbought and oversold levels.
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data['RSI'], 'orange', label='RSI')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.legend(loc='best')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    return fig

# Add RSI graph to the app
st.subheader('Relative Strength Index (RSI)')
st.pyplot(plot_rsi_graph((15, 6), stock_data))


# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame for plotting predictions
plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_predictions.reshape(-1)
    },
    index=stock_data.index[splitting_len + 100:]
)

st.subheader("Original values vs Predicted values")
st.write(plotting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(stock_data.Close, 'b', label="Data - not used")
plt.plot(plotting_data['original_test_data'], 'orange', label="Original Test Data")
plt.plot(plotting_data['predictions'], 'green', label="Predicted Test Data")
plt.legend(loc='best')
plt.title('Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig)

# Calculate accuracy metrics
mse = mean_squared_error(plotting_data['original_test_data'], plotting_data['predictions'])
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(plotting_data['original_test_data'], plotting_data['predictions'])

# Display metrics in Streamlit
st.subheader('Prediction Accuracy Metrics')

with st.expander(f"**Mean Squared Error (MSE):** {mse:.2f}"):
    # st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"MSE of {mse:.2f} means the average of the squared differences between the predicted and actual values is {mse:.2f}.")
    st.write("A lower MSE indicates a better fit of the model to the data, as it means that the predicted values are closer to the actual values.")
    st.write("However, MSE is sensitive to outliers, meaning that larger errors are penalized more heavily than smaller ones because the errors are squared.")

with st.expander(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}"):
    # st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"RMSE of {rmse:.2f} means that, on average, the predictions are off by about **${rmse:.2f}**.")
    st.write("This metric gives more weight to larger errors, meaning that large deviations from the actual values will influence the result more.")
    st.write("Lower RMSE values indicate better model performance, and this value gives you an idea of how much the predicted values deviate from the actual values.")

with st.expander(f"**Mean Absolute Error (MAE):** {mae:.2f}"):
    st.write(f"MAE of {mae:.2f} means that, on average, the model's predictions are off by **${mae:.2f}**.")
    st.write("This metric treats all errors equally and gives a more balanced measure of the overall prediction accuracy.")
    st.write("MAE is a simple and intuitive metric that represents the average of the absolute differences between the predicted and actual values.")
    st.write("Unlike MSE, MAE does not penalize larger errors more heavily, making it less sensitive to outliers. It's a good indicator of the overall prediction accuracy without being influenced by large errors.")

# Percentage-based accuracy
def calculate_percentage_accuracy(actual, predicted, tolerance=0.05):
    """
    Calculate the percentage of predictions within a given tolerance of the actual values.
    :param actual: numpy array or pandas Series, the actual values.
    :param predicted: numpy array or pandas Series, the predicted values.
    :param tolerance: float, the tolerance as a fraction (e.g., 0.05 for 5%).
    :return: float, the percentage of predictions within the tolerance.
    """
    within_tolerance = np.abs(predicted - actual) <= (tolerance * actual)
    accuracy_percentage = np.mean(within_tolerance) * 100  # Convert to percentage
    return accuracy_percentage

# Calculate percentage-based accuracy
accuracy_5_percent = calculate_percentage_accuracy(
    plotting_data['original_test_data'], plotting_data['predictions'], tolerance=0.05
)
accuracy_10_percent = calculate_percentage_accuracy(
    plotting_data['original_test_data'], plotting_data['predictions'], tolerance=0.10
)

# Display percentage accuracy
st.write(f"**Accuracy within ±5%:** {accuracy_5_percent:.2f}%")
st.write(f"**Accuracy within ±10%:** {accuracy_10_percent:.2f}%")
