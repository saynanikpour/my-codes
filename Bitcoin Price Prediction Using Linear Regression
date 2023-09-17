import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from persiantools.jdatetime import JalaliDate
from datetime import datetime

df = pd.read_csv("BTC-USD.csv")

df["Benefit"] = df["Close"] - df["Open"]
df["Jalali"] = df["Date"].apply(lambda d: JalaliDate(datetime.strptime(d, "%Y-%m-%d")))
df["Close_7_Days_Later"] = df["Close"].shift(-7)
df["Year"] = df["Jalali"].apply(lambda d: d.year)
df["Month"] = df["Jalali"].apply(lambda d: d.month)
df["Weekday"] = df["Jalali"].apply(lambda d: d.isoweekday())

df.drop("Jalali", axis=1, inplace=True)


df.dropna(inplace=True)

X = np.vstack((df["Benefit"].values, df["Year"].values, df["Weekday"].values, df["Month"].values)).T
Y = df["Close_7_Days_Later"].values

model = LinearRegression()
model.fit(X, Y)

predict = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(df.index, Y, label='True')
plt.plot(df.index, predict, label='Predict', linestyle='--', color='purple')
plt.title('Close Price for 7 days later')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

predicted_prices_7_days_later = predict.tolist()
print("Close Price for 7 days later:")
print(predicted_prices_7_days_later)
