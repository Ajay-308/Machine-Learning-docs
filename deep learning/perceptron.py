#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv(r"C:\Users\mraja\Downloads\placement.csv")


# In[7]:


df.shape


# In[8]:


df.sample(5)


# In[13]:


custom_palette = {0: 'black', 1: 'green'}
sns.scatterplot(df['cgpa'], df['resume_score'], hue=df['placed'], palette=custom_palette)


# In[14]:


X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[15]:


from sklearn.linear_model import Perceptron
p = Perceptron()


# In[16]:


p.fit(X,y)


# In[17]:


p.coef_


# In[18]:


p.intercept_


# In[21]:



get_ipython().system('pip install mlxtend')

from mlxtend.plotting import plot_decision_regions


# In[24]:


plot_decision_regions(X.values, y.values, clf=p, legend=2)


# In[ ]:




