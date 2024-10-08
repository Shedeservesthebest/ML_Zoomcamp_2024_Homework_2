#!/usr/bin/env python
# coding: utf-8

# In[647]:


import pandas as pd
df = pd.read_csv('laptops.csv')
#df.head()


# ## Preparing the dataset

# In[648]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
#df
#df.info()


# In[649]:


RSSF = df[['ram','storage', 'screen', 'final_price']]
#RSSF


# ## EDA

# In[650]:


import matplotlib.pyplot as plt
import seaborn as sns
Final = df['final_price']
#Final.plot(kind='hist', bins=30)

sns.histplot(Final[Final<7000], bins=30)


#plt.show()


# Final_price has a long tail.

# In[651]:


log_price = np.log1p(df.final_price)

plt.figure(figsize=(6, 4))

sns.histplot(log_price, bins=40, alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log tranformation')

plt.show()


# ## Question 1
# 
# 

# There's one column with missing values. What is it?

# In[652]:


df.isnull().sum()


# Screen

# ## Question 2

# What's the median (50% percentile) for variable 'ram'?

# In[653]:


df['ram'].median()


# ## Question 3

# ## Prepare and split the dataset

# Shuffle the dataset (the filtered one you created above), use seed 42.
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
# Use the same code as in the lectures

# In[654]:


n = len(RSSF)


# In[655]:


n_val = int(n*0.2)
n_test = int (n*0.2)
n_train = int(n - (n_val + n_test))


# In[656]:


n_train


# In[657]:


n, n_val + n_test + n_train


# In[659]:


df_train =RSSF.iloc[ : n_train]
df_val = RSSF.iloc[n_train : n_train + n_val]
df_test =RSSF.iloc[n_train + n_val : ]


# In[660]:


import numpy as np
idx = np.arange(n)
df_train = RSSF.iloc[idx[ : n_train]]
df_val = RSSF.iloc[idx[n_train : n_train + n_val]]
df_test = RSSF.iloc[idx[n_train + n_val : ]]


# In[661]:


np.random.seed(42)
len(df_train),len(df_val),len(df_test)


# In[662]:


#To make the index fixed
Train_data = df_train.reset_index(drop = True)
Val_data = df_val.reset_index(drop = True)
Test_data = df_test.reset_index(drop = True)


# In[663]:


y_train = np.log1p(Train_data.final_price.values)
y_val = np.log1p(Val_data.final_price.values)
y_test = np.log1p(Test_data .final_price.values)


# In[664]:


#To remove the column from all the inputs
#To also avoid using it when making the prediction of the final_price
del Train_data['final_price']
del Val_data['final_price']
del Test_data['final_price']


# In[665]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y_train)
    
    return w[0], w[1:]


# In[666]:


base =['ram','storage', 'screen']


# With 0

# In[667]:


def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[668]:


X_train = prepare_X(Train_data)
w_0, w = train_linear_regression(X_train, y_train)


# In[669]:


y_pred = w_0 + X_train.dot(w)


# In[670]:


y_pred


# In[671]:


rmse(y_train, y_pred)


# In[672]:


X_val = prepare_X(Val_data )
y_pred = w_0 + X_val.dot(w)


# In[673]:


rmse(y_val, y_pred).round(2)


# with the mean

# In[674]:


df['screen'].mean()


# In[678]:


def prepare_X(RSSF):
    df_num = RSSF[base]
    df_num = df_num.fillna(15.168112244897959)
    X = df_num.values
    return X


# In[679]:


X_train = prepare_X(Train_data)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
y_pred


# In[680]:


X_val = prepare_X(Val_data )
y_pred = w_0 + X_val.dot(w)

rmse(y_val, y_pred).round(2)


# ## Question 4

# In[681]:


def prepare_X(RSSF):
    df_num = RSSF[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# In[682]:


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# In[683]:


X_train = prepare_X(Train_data)
X_val = prepare_X(Val_data)

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' %r, rmse(y_val, y_pred))


# r = 0 gave the best model

# In[684]:


X_train = prepare_X(Train_data)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0)

X_val = prepare_X(Val_data)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))

X_test = prepare_X(Test_data)
y_pred = w_0 + X_test.dot(w)
print('test:', rmse(y_test, y_pred))



# ## Question 5

# In[686]:


import numpy as np

# Define the seeds to try
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# List to store the RMSE scores for each seed
rmse_scores = []

# Helper function to compute RMSE
def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

# Function to prepare X with the base features and fill missing values
def prepare_X(RSSF):
    df_num =RSSF[base]
    df_num = df_num.fillna(0)  # Fill missing values with 0
    X = df_num.values
    return X

# Function to train linear regression model
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])  # Add bias term
    X = np.column_stack([ones, X])  # Add bias as a feature
    XTX = X.T.dot(X)  # X^T * X
    XTX_inv = np.linalg.inv(XTX)  # Inverse of X^T * X
    w = XTX_inv.dot(X.T).dot(y)  # Compute weights
    return w[0], w[1:]  # Return bias term and feature weights

# Loop through each seed and evaluate the model
for seed in seeds:
    # Shuffle the data with the current seed
    np.random.seed(seed)
    idx = np.arange(len(RSSF))
    np.random.shuffle(idx)

    # Perform the 60%/20%/20% split
    n = len(df)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    n_test = n - (n_train + n_val)

    df_train = RSSF.iloc[idx[:n_train]]
    df_val =RSSF.iloc[idx[n_train:n_train + n_val]]
    df_test = RSSF.iloc[idx[n_train + n_val:]]

    # Reset the index
    Train_data = df_train.reset_index(drop=True)
    Val_data = df_val.reset_index(drop=True)
    Test_data = df_test.reset_index(drop=True)

    # Prepare the target variable and remove it from the features
    y_train = np.log1p(Train_data.final_price.values)
    y_val = np.log1p(Val_data.final_price.values)
    y_test = np.log1p(Test_data.final_price.values)

    del Train_data['final_price']
    del Val_data['final_price']
    del Test_data['final_price']

    # Prepare input matrices for training
    X_train = prepare_X(Train_data)
    X_val = prepare_X(Val_data)

    # Train the model without regularization
    w_0, w = train_linear_regression(X_train, y_train)

    # Make predictions on the validation set
    y_pred_val = w_0 + X_val.dot(w)

    # Compute RMSE for the validation set and store it
    rmse_val = rmse(y_val, y_pred_val)
    rmse_scores.append(rmse_val)

# Compute the standard deviation of the RMSE scores
std_rmse = np.std(rmse_scores)

# Print the rounded standard deviation
print("Standard deviation of RMSE scores:", round(std_rmse, 3))


# ## Question 6

# In[687]:


import numpy as np

# Use seed 9 for splitting the data
seed = 9
np.random.seed(seed)

# Shuffle the data
idx = np.arange(len(RSSF))
np.random.shuffle(idx)

# Perform the 60%/20%/20% split
n = len(RSSF)
n_train = int(n * 0.6)
n_val = int(n * 0.2)
n_test = n - (n_train + n_val)

df_train = RSSF.iloc[idx[:n_train]]
df_val = RSSF.iloc[idx[n_train:n_train + n_val]]
df_test = RSSF.iloc[idx[n_train + n_val:]]

# Combine training and validation sets
df_combined_train = pd.concat([df_train, df_val], ignore_index=True)

# Reset the index of the test set
Test_data = df_test.reset_index(drop=True)

# Prepare the target variable (log transformation) and remove it from the features
y_combined_train = np.log1p(df_combined_train.final_price.values)
y_test = np.log1p(Test_data.final_price.values)

del df_combined_train['final_price']
del Test_data['final_price']

# Function to prepare X with base features and fill missing values
def prepare_X(RSSF):
    df_num =RSSF[base]
    df_num = df_num.fillna(0)  # Fill missing values with 0
    X = df_num.values
    return X

# Prepare input matrices
X_combined_train = prepare_X(df_combined_train)
X_test = prepare_X(Test_data)

# Train ridge regression model with r = 0.001 (L2 regularization)
def train_ridge_regression(X, y, r):
    ones = np.ones(X.shape[0])  # Add bias term
    X = np.column_stack([ones, X])  # Add bias as a feature

    XTX = X.T.dot(X)  # X^T * X
    XTX_reg = XTX + r * np.eye(XTX.shape[0])  # Add L2 regularization (r * I)
    XTX_inv = np.linalg.inv(XTX_reg)  # Inverse of regularized X^T * X
    w = XTX_inv.dot(X.T).dot(y)  # Compute weights
    return w[0], w[1:]  # Return bias term and feature weights

# Set regularization parameter r = 0.001
r = 0.001

# Train the model
w_0, w = train_ridge_regression(X_combined_train, y_combined_train, r)

# Make predictions on the test set
y_pred_test = w_0 + X_test.dot(w)

# Compute RMSE for the test set
def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

rmse_test = rmse(y_test, y_pred_test)

# Print the RMSE for the test set
print("Test RMSE:", rmse_test)


# In[ ]:




