#!/usr/bin/env python
# coding: utf-8

# ## Name : Utkarsha Kale

# ## Task No : 01 Prediction using supervised ML

# ## Importing all required libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import data

# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(10)


# ## Plotting the distribution of scores

# Declare dependent and independent variables

# In[3]:


x=data['Hours']
y=data['Scores']


# In[4]:


plt.scatter(x,y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# ## Preparing the data

# In[5]:


X =x.values.reshape(-1,1) 


# In[6]:


X.shape


# ## Splitting of data 

# Now that we have our dependent and Independent, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# In[8]:


print(f"Rows in train set: {len(y_train)}\nRows in test set: {len(y_test)}\n")


# ## Training the Model

# In[9]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")


# Plotting the regression line for testing a data. 

# In[10]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line,c="orange");
plt.show()


# ## Making Predictions

# In[11]:


print(X_test) 
y_pred = regressor.predict(X_test) 


# Comparing Actual vs Predicted

# In[12]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# you can also test with your own data

# In[13]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaluating the model

# This step is particularly important to compare how well different algorithms perform on a particular dataset.
# 
# 

# In[14]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 


# conclusion: prediction using supervised machine learning task was successfully exected and predicted score if student study for 9.25hr/day is 93.69173248737538

# ## Thank You
