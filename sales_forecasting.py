
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import os

df=pd.read_csv("data/retail_sales.csv")
print(df.head())
print(df.columns)

df['data']=pd.to_datetime(df['data'])
df=df.sort_values('data')

plt.figure(figsize=(10,6))
plt.plot(df['data'], df['venda'], label="Sales Over Time", color="blue")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend")
plt.legend()
plt.show()

train_size=int(len(df) * 0.8)
train, test=df['venda'][:train_size], df['venda'][train_size:]

model=ARIMA(train, order=(5,1,0))   
model_fit=model.fit()

forecast=model_fit.forecast(steps=len(test))
print("Forecast values:",forecast)

plt.figure(figsize=(10,6))
plt.plot(df['data'][train_size:],test,label="Actual Sales",color="blue")
plt.plot(df['data'][train_size:],forecast,label="Forecasted Sales",color="red")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()

mse=mean_squared_error(test, forecast)
print("Mean Squared Error:", mse)

os.makedirs("output",exist_ok=True)
df.to_csv("output/sales_analysis.csv",index=False)

