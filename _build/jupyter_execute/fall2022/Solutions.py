#!/usr/bin/env python
# coding: utf-8

# # Solutions
# 
# Example solutions for challenge exercises from each chapter in this book.

# # Chapter 1

# In[ ]:





# # Chapter 2

# In[ ]:





# # Chapter 3

# In[ ]:





# # Chapter 4

# In[ ]:





# # Chapter 5

# In[ ]:





# # Chapter 6

# In[ ]:





# # Chapter 7

# 1. Compare our "by hand" OLS results to those producd by sklearn's `LinearRegression` function. Are they the same? 
#     * Slope = 4
#     * Intercept = -4
#     * RMSE = 2.82843
#     * y_hat = y_hat = B0 + B1 * data.x

# In[1]:


# Recreate dataset
import pandas as pd
data = pd.DataFrame({"x": [1,2,3,4,5],
                     "y": [2,4,6,8,20]})
data


# In[2]:


# Our "by hand" OLS regression information:
B1 = 4
B0 = -4
RMSE = 2.82843
y_hat = B0 + B1 * data.x


# In[3]:


# use scikit-learn to compute R-squared value
from sklearn.linear_model import LinearRegression

lin_mod = LinearRegression().fit(data[['x']], data[['y']])
print("R-squared: " + str(lin_mod.score(data[['x']], data[['y']])))


# In[4]:


# use scikit-learn to compute slope and intercept
print("scikit-learn slope: " + str(lin_mod.coef_))
print("scikit-learn intercept: " + str(lin_mod.intercept_))


# In[5]:


# compare to our by "hand" versions. Both are the same!
print(int(lin_mod.coef_) == B1)
print(int(lin_mod.intercept_) == B0)


# In[6]:


# use scikit-learn to compute RMSE
from sklearn.metrics import mean_squared_error

RMSE_scikit = round(mean_squared_error(data.y, y_hat, squared = False), 5)
print(RMSE_scikit)


# In[7]:


# Does our hand-computed RMSE equal that of scikit-learn at 5 digits?? Yes!
print(round(RMSE, 5) == round(RMSE_scikit, 5))


# 2. 

# In[ ]:





# In[ ]:





# In[ ]:





# 3.

# In[ ]:





# In[ ]:





# In[ ]:





# # Chapter 8

# In[ ]:





# # Chapter 9

# In[ ]:





# In[ ]:




