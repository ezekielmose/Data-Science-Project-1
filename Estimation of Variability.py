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


# In[1]:


from statistics import variance


# In[6]:


import pandas as pd


# In[7]:


dataset2= pd.read_excel("D:\Studies\Data Science\Tokyo 2021 Dataset\\Medals.xlsx")


# In[9]:


variance(dataset2["Gold"])


# In[11]:


variance(dataset2["Total"])


# In[12]:


#Standard Variation

from statistics import stdev


# In[13]:


stdev(dataset2["Gold"])


# In[14]:


#Mean absolute Daviation

from numpy import mean, absolute


# In[15]:



mean(absolute(dataset2["Gold"]-mean(dataset2["Gold"])))


# In[16]:


#Median absolute Daviation

from numpy import median

median(absolute(dataset2["Gold"]- median(dataset2["Gold"])))


# In[19]:


# interquatile range
import numpy as np

q3, q1= np.percentile(dataset2["Gold"], [75,25])
iqr= q3-q1
iqr


# In[27]:


#Exploqaring data distribution
#Boxplot
import matplotlib.pyplot as plt

dataset2.boxplot(column="Total", by=None, ax=None)


# In[29]:


#Frequency Table
frequence_table= dataset2.copy()
frequence_table["freq"]=pd.cut(dataset2["Gold"], 10)
frequence_table.groupby("freq")["Rank"].count().reset_index()


# In[44]:


#Histogram

import matplotlib.pyplot as plt

plt.hist(x=dataset2["Total"], bins='auto', color='#0504aa', 
         alpha=0.7, rwidth=0.85 )


# In[52]:


#Desinty Plot

ax= dataset2["Gold"].plot.hist(density=True, xlim=[1,12], bins=range(1,12)) 
dataset2["Gold"].plot.density(ax=ax)


# In[53]:



#Binary and Categorical data

dataset2["Gold"].mode()


# In[ ]:




