#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MahreenAthar/MLOps-Assignment-2/blob/BranchMahreen/Model_StockExchange.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Importing Libraries

# In[20]:


# !pip3 install yfinance
# !pip3 install Flask


# In[21]:


import pandas as pd
import numpy as np
import datetime
import os
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import yfinance as yf


# In[22]:


app = Flask(__name__, template_folder='/templates')


# # Loading the dataset

# In[23]:


ticker = yf.Ticker("GOOG")
data = ticker.history(start=datetime.date(2019, 1, 1), end=datetime.date.today())
data.reset_index(inplace=True)


# # Splitting Data
# 80% training, 20% testing

# In[24]:


train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]


# # Define input features 
# Open, High, Low, and Volume

# In[25]:


X_train = train_data[['Open', 'High', 'Low', 'Volume']]
X_test = test_data[['Open', 'High', 'Low', 'Volume']]


# # Define target variable (Close)

# In[26]:


y_train = train_data['Close']
y_test = test_data['Close']


# # Create polynomial features of degree 2

# In[27]:


poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# # Train polynomial regression model

# In[28]:


model = LinearRegression()
model.fit(X_train_poly, y_train)


# # Print mean squared error on test data
# 

# In[29]:


y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)


# # Save the trained model

# In[30]:


joblib.dump(model, 'model.pkl')


# # Load the trained model

# In[31]:


model = joblib.load('model.pkl')


# In[32]:


@app.route('/')
def chart():
    # Extract data for charting
    chart_data = data[['Date', 'High', 'Low']].values.tolist()

    # Make predictions on test data and compute evaluation metrics
    y_pred = model.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Render the template with the chart data and evaluation metrics
    return render_template('Index.html', chart_data=chart_data, r2=r2, mse=mse, mae=mae)


# In[33]:


@app.route('/interface')
def interface():
    # Render the input template
    return render_template('Interface.html')

@app.route("/forecast", methods=['POST'])
def forecast():
    try:
        # Extract the input features from the form
        Open = float(request.form['Open'])
        Max = float(request.form['Max'])
        Min = float(request.form['Min'])
        Volume = float(request.form['Volume'])

        # Create a DataFrame with the input features and predict the output
        d = {'Open': [Open], 'High': [Max], 'Low': [Min], 'Volume': [Volume]}
        d = pd.DataFrame(d)
        result = model.forecast(poly.fit_transform(d))

        # Render the prediction template with the result
        return render_template('Forecast.html', prediction=result[0], output="")
    except:
        # Render the prediction template with an error message if the input features are invalid
        return render_template('Forecast.html', prediction=0, output="Enter input features in the previous page")


# In[ ]:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=False)


# In[ ]:




