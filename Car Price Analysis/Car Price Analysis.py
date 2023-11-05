#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[8]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


car=pd.read_csv('C:/Users/aryan/Desktop/Dataset/Car Price Analysis.csv')


# In[10]:


car.head()


# In[11]:


car.info()


# In[12]:


# how many affordable cars
car[car['Affordable']=='YES'].count()


# In[13]:


sns.countplot(x='Affordable',data=car)


# In[14]:


sns.pairplot(car)


# In[15]:


sns.displot(car['price'],kde=True)


# In[16]:


sns.heatmap(car.corr(),annot=True)


# In[17]:


sns.barplot(x='Affordable',y='price',data=car)


# In[18]:


sns.boxplot(x='Affordable',y='price',data=car)


# In[19]:


# top 5 expensive cars
car.sort_values('price',ascending=False).head(5)


# In[20]:


# top 5 least expensive cars
car.sort_values('price',ascending=True).head(5)


# In[21]:


sns.lmplot(x='price',y='carlength',data=car,hue='Affordable')


# In[22]:


sns.lmplot(x='price',y='carwidth',data=car,hue='Affordable')


# In[23]:


sns.lmplot(x='price',y='carheight',data=car,hue='Affordable')


# In[24]:


car.columns


# In[25]:


# training
X=car[['carlength', 'carwidth', 'carheight']]
y=car['price']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=47)


# In[28]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[29]:


lm.intercept_


# In[30]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[31]:


predictions=lm.predict(X_test)


# In[32]:


plt.scatter(y_test,predictions)


# In[34]:


sns.histplot((y_test-predictions),kde=True,bins=50)


# In[35]:


from sklearn import metrics


# In[36]:


metrics.mean_absolute_error(y_test,predictions)


# In[37]:


metrics.mean_squared_error(y_test,predictions)


# In[38]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[39]:


metrics.explained_variance_score(y_test,predictions)


# In[40]:


metrics.r2_score(y_test,predictions)


# # 86.08% of variance is explained by the model
