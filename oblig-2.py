#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Predicting stock prices is a difficult endeavor
# that incorporates many variables. 

# Stock price movements are highly stochastic,
# however regression can be used. 


# At first, we considered linear regression. 

# but forecast stock values, particularly for specific days

# can be difficult because stock prices are frequently complex.





# In[59]:


#linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the CSV data from the path using Pandas
df = pd.read_csv('C:\\Users\\Eier\\TSLA.csv') 

# Date in df is stored as a string so we need to convert it to DateTime input
df['Date'] = pd.to_datetime(df['Date'])

#input is Date=> x=Date 
#outputis price=> y=Close Price
# linear regression works with numerical features so 
# we need to convet the DateTime to timestamps which is numerical 
X = df['Date'].apply(lambda x: x.timestamp())  
X = X.values.reshape(-1, 1)
Y = df['Close'].values.reshape(-1, 1)


#create a linear regression model
linear_regressor = LinearRegression()
 # perform linear regression
linear_regressor.fit(X, Y) 
# make predictions
Y_pred = linear_regressor.predict(X)  



# Calculate R-squared value to find the percentage score
r2 = r2_score(Y, Y_pred)

# Print the R-squared value
print(f"R-squared (Prediction Percentage Score): {r2 * 100:.2f}%")


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Date (Timestamp)')
plt.ylabel('Close Price')
plt.show()




# In[ ]:





# In[66]:


# Nonlinear regression, such as polynomial regression, is more appropriate.

# since it contains polynumial terms such as quadratic.

# This allows the model to suit curvilinear patterns better.


df['Date'] = pd.to_datetime(df['Date'])


X = df['Date'].apply(lambda x: x.timestamp())  
X = X.values.reshape(-1, 1)
Y = df['Close'].values.reshape(-1, 1)


# Choosing the degree of the polynomial
poly_degree = 4 

# Create polynomial features
poly_features = PolynomialFeatures(degree=poly_degree)
X_poly = poly_features.fit_transform(X)

# Fit the polynomial Model
model = LinearRegression()
model.fit(X_poly, y)

# Predict Y values using the model
y_pred = model.predict(X_poly)

# Calculate R-squared value to find the percentage score
r2 = r2_score(Y, Y_pred)

# Print the R-squared value
print(f"R-squared (Prediction Percentage Score): {r2 * 100:.2f}%")

# Plot the results
plt.scatter(X, y, label='Data (Timestamp)', color='blue')
plt.plot(X, y_pred, label='Polynomial Regression', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()




# In[ ]:




