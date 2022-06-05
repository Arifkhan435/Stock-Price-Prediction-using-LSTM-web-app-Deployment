##Importing necessary libraries
import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2021-12-31'

st.title("Stock Price Prediction")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

##describing data
st.subheader('Data from 2010 - 2021')
st.write(df.describe())

##Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'b')
plt.plot(df.Close,'g')
st.pyplot(fig)


##splitting data into training and testing
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

data_train_array = sc.fit_transform(data_train)

## Load the trained Model
model = load_model('stock_prediction.h5')

### Testing Part
past_100_days = data_train.tail(100)
##append last 100 data to test data
final_df = past_100_days.append(data_test, ignore_index=True)

input_data = sc.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

## converting x_test and y_test into array
x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

##scaling factor
scaler = sc.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor


##Final Plot

st.subheader('Predictions vs Actual')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Actual Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)















