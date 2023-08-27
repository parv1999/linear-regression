#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data  = pd.read_csv('./datasets/kc_house_data.csv')


# In[2]:


data.info()


# In[3]:


data.drop(columns=['id'],inplace=True)
data.describe()


# In[4]:


plt.figure(figsize=(17,12))
sns.heatmap(data.corr(),cmap='RdPu',annot=True)
plt.show()


# In[5]:


df = data.drop(columns=['date','zipcode','sqft_living15','sqft_lot15','sqft_living','sqft_lot','sqft_above','waterfront'],inplace=False)


# In[6]:


data[['zipcode','waterfront','view','grade','condition']].nunique()


# In[7]:


from sklearn.utils import shuffle


# In[8]:


df = data.drop(columns=['date','zipcode','grade','sqft_living','sqft_lot','sqft_above'],inplace=False)


# In[9]:


df = shuffle(df)

df_test  = df.sample(frac = 0.20)

df_train = df.drop(df_test.index)


# In[10]:



x_train_ = np.array( df_train.drop(columns=['price'],axis=1) ) 
y_train_ = np.array( df_train['price'].values )

x_test_  = np.array( df_test.drop(columns=['price'],axis=1) )
y_test_  = np.array( df_test['price'].values)


# In[11]:


x_train = ( x_train_ - np.mean(x_train_) ) / np.std(x_train_ )
y_train = ( y_train_ - np.mean(y_train_) ) / np.std(y_train_ )


# In[12]:


# np.dot(x_train.T,predictions-y).shape
m=np.dot(x_train.T,predictions-y_train)
m.shape


# In[16]:


w         = np.full((x_train.shape[1],1),1)
b         = 1

j,epochs  = [],[]
cost=0
def yhat(x,w,b) :
    
    return np.dot(x,w) + b 

def calc_cost( predictions,y ) : 
    
    return sum( (predictions-y)**2 )/(2*y.shape[0])


def gradescent(w,b,x,predictions,y,alpha) :
    
    epoch = 0 
    temp  = -np.inf
    cost  = calc_cost(predictions,y)
    
    predictions  = predictions.reshape((x_train.shape[0],1))
    y            = y.reshape((x_train.shape[0],1))
    while cost > temp :
          cost        = calc_cost( predictions , y )
          j           .append(cost)
          epochs      .append(epoch)
          w           = w - alpha*np.dot(x.T,predictions-y)/x.shape[0]
          b           = b - alpha*np.sum(predictions - y)/x.shape[0]
          predictions = yhat(x,w,b).reshape((x_train.shape[0],1))
          temp        = calc_cost(predictions,y)  
          print(temp)  
          epoch       = epoch + 1  
    return w,b

predictions = yhat(x_train,w,b)
W,B         = gradescent(w,b,x_train,predictions,y_train,0.01)

plt.plot(epochs,j,'bo')


# In[ ]:


# calc_cost(predictions,y_train)


# In[18]:


# a = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
# c = np.array([1,2,3,4])

# np.dot(a,c)
# predictions = yhat()


# In[ ]:


# np.dot((predictions-y).T,x_train)
# k         = np.ones((x_train.shape[1],1))
# k.shape

# yhat(x_train,w,b)
# predictions.reshape(x_train.shape[0],1).shape
# np.dot(x_train,predictions-y)
# y = y_train
# y.reshape(x_train.shape[0],1)
# np.sum(np.dot((predictions-y).T,x_train)>0)
# np.dot((predictions-y_train).T,x_train)
# (predictions-y).shape


# In[ ]:


# np.dot(x_train,predictions-y_train)
# x = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1]])
# w = np.array([1,1,1,1]).T # dims = n x 1
# y_train.shape
# y=np.dot(x,w)
# predictions = predictions.reshape(x_train.shape[0],1)
# predictions.shape


# In[ ]:


# y_train = y_train.reshape((x_train.shape[0],1))


# In[ ]:


# (predictions-y_train).shape


# In[ ]:


# x_train.shape , predictions.shape


# In[ ]:


# w.shape# y = np.array([20,24,3])

# pred = yhat(x,w,0)

# calc_cost(pred,y)


# In[ ]:


# n=[1,2,3]
# n .append(4)
# n


# In[ ]:


# np.ones(x_train.shape[1])


# In[ ]:





# In[ ]:




