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


# In[5]:



import numpy as np
import pandas as pd
dataset=pd.read_excel("D:\Studies\Data Science\Projects\Project1\Tokyo 2021 Dataset\\Medals.xlsx")
q1=np.percentile(dataset["Total"], 75)


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


# In[12]:


#Desinty Plot

ax= dataset["Gold"].plot.hist(density=True, xlim=[1,18], bins=range(1,18)) 
dataset["Gold"].plot.density(ax=ax)


# In[53]:



#Binary and Categorical data

dataset2["Gold"].mode()


# In[3]:


import pandas as pd
dataset= pd.read_excel("D:\Studies\Data Science\Projects\Project1\Tokyo 2021 Dataset\\Medals.xlsx")


# In[4]:


dataset.head()


# In[7]:


#Exploaring binary and categorigal data
dataset["Total"].mode()


# In[11]:


#Bar Graph

import matplotlib.pyplot as plt
ax=dataset.iloc[:10,:].plot.bar(x="Team/NOC", y="Gold", figsize=(4,4), legend= False)
ax.set_xlabel("Gold County")
ax.set_ylabel("Country")


# In[26]:


#pie chart
pie_data= dataset.iloc[:10, :]
plt.pie(pie_data["Gold"], labels=pie_data["Team/NOC"], autopct='%1.1f')


# In[27]:


#Correlation
dataset.corr()


# In[30]:


#scatterplot
dataset.plot.scatter(x="Silver", y="Bronze")


# In[39]:


#Hexagonal binning/plot
dataset.plot.hexbin(x="Silver",y="Bronze",gridsize=20, sharex=False)


# In[42]:


#contour plot

import seaborn as sns
ax=sns.kdeplot(dataset["Gold"], dataset["Silver"])
ax


# In[43]:


#heatmap
sns.heatmap(dataset.corr())


# In[51]:


#contigency table
crosstab=dataset.pivot_table(index='Team/NOC', columns='Gold', aggfunc=lambda x:len(x),margins=True)
crosstab.fillna(0)


# In[52]:


#Violin Plot
sns.violinplot(x=dataset["Gold"])


# In[ ]:




