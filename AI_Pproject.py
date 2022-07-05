#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


dataset=pd.read_csv('breast-cancer.csv')


# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[6]:


# loading the data to a data frame
dataset = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[7]:


dataset['label'] = breast_cancer_dataset.target


# In[8]:


dataset.head()


# In[9]:


dataset.info()


# In[10]:


dataset.tail()


# In[11]:


import matplotlib.pyplot as plt
dataset.hist(figsize = (16,18))


# In[12]:


#1 --> Benign

#0 --> Malignant 
dataset['label'].value_counts()


# In[13]:


dataset.info()


# In[14]:


#1 --> Benign

#0 --> Malignant 
dataset.groupby('label').mean()


# In[15]:


x = dataset.drop(columns='label', axis=1)
y = dataset['label']


# In[16]:


# Normalizing data using "StandardScaler"
scaler=StandardScaler()
scaler.fit(dataset)
dataset=scaler.transform(dataset)


# In[17]:


#spiliting data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[18]:


#model  of project
model=LogisticRegression()


# In[19]:


model.fit(x_train,y_train)


# In[20]:


print('Accuracty of train is :', model.score(x_train,y_train)*100)


# In[21]:


print('Accuracty of test is:', model.score(x_test,y_test)*100)


# In[22]:


# Testing model
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

input_data_as_numpy_array = np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

classify = model.predict(input_data_reshaped)
print(classify)

if (classify[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


# In[23]:


# Testing model
input_data=(20.13,28.25,131.2,1261,0.0978,0.1034,0.144,0.09791,0.1752,0.05533,0.7655,2.463,5.203,99.04,0.005769,0.02423,0.0395,0.01678,0.01898,0.002498,23.69,38.25,155,1731,0.1166,0.1922,0.3215,0.1628,0.2572,0.06637)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

classify = model.predict(input_data_reshaped)
print(classify)

if (classify[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


# In[ ]:




