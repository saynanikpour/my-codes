import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


data = pd.read_csv('BTC-USD.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.dropna()


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scores = []

for train_index, test_index in kf.split(data):
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

    X_train = train_data[['Open', 'High', 'Low', 'Volume']]
    y_train = train_data['Close']

    X_test = test_data[['Open', 'High', 'Low', 'Volume']]
    y_test = test_data['Close']

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)


    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    predictions = model.predict(X_test_imputed)

    mse = mean_squared_error(y_test, predictions)
    mse_scores.append(mse)

    plt.figure(figsize=(8, 6))
    plt.plot(test_data.index, y_test, label='True')
    plt.plot(test_data.index, predictions, label='predict', linestyle='--')
    plt.title('predict')
    plt.xlabel('Date')
    plt.ylabel('BTC price')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'error MSE: {mse}')
    print('-' * 40)

mse_scores = np.array(mse_scores)

rmse_scores = np.sqrt(mse_scores)

plt.figure(figsize=(8, 6))
plt.plot(range(1, n_splits + 1), rmse_scores, marker='o', linestyle='-', color='b')
plt.title('error RMSE for each section (KFold)')
plt.xlabel('sections')
plt.ylabel('error RMSE')
plt.grid(True)
plt.show()

average_rmse = rmse_scores.mean()
print(f'mean RMSE for each section {average_rmse}')
