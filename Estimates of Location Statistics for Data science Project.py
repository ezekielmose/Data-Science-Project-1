#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


dataset= pd.read_excel("D:\Studies\Data Science\Tokyo 2021 Dataset\\Medals.xlsx")


# In[8]:


dataset.head()


# In[9]:


dataset.tail()


# In[10]:


#mean
dataset["Total"].mean()


# In[11]:


dataset["Bronze"].mean()


# In[12]:


dataset["Gold"].mean()


# In[13]:


import numpy as np


# In[14]:


#weghted mean

np.average(dataset["Gold"], weights=dataset["Total"])


# In[15]:


np.average(dataset["Total"], weights=dataset["Gold"])


# In[16]:


np.average(dataset["Bronze"], weights=dataset["Total"])


# In[17]:


#trimed mean

from scipy.stats import trim_mean
trim_mean(dataset["Gold"], 0.1)


# In[18]:


list(dataset["Gold"])


# In[20]:


dataset["Total"].median()


# In[21]:


q1=np.percentile(dataset["Total"], 75)


# In[22]:


q1


# In[24]:


q1=np.percentile(dataset["Total"], 25)
q1


# In[25]:


q1=np.percentile(dataset["Total"], 50)
q1


# In[ ]:




