#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER PREDICTION

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[4]:


data.head()


# In[5]:


data.count()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


#removeing null and unwanted col
data.columns
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)


# In[9]:


data.shape


# In[10]:


type(data.columns)


# In[11]:


#coverting feature into list 
l=list(data.columns)


# In[12]:


#grouping similar feature into same group
feature_mean= l[1:11]
feature_se= l[11:21]
feature_worst= l[21:]


# In[13]:


feature_mean


# In[14]:


#checking y variable counts
data['diagnosis'].value_counts()


# # Explore the data

# In[15]:


data.describe()


# In[16]:


sns.countplot(data['diagnosis'])
data['diagnosis'].value_counts()


# In[17]:


#correlation plot
corr = data.corr()


# In[18]:


plt.figure(figsize=(8,8))
sns.heatmap(corr);


# In[19]:


#converting y variable into 0s &1s
data['diagnosis']=data['diagnosis'].map({"M":1,"B":0})


# In[20]:


data['diagnosis'].value_counts()


# In[21]:


type(data['diagnosis'])


# In[22]:


#dividing data into independent and dependent variable
X = data.drop('diagnosis',axis = 1)
Y = data['diagnosis']


# In[23]:


X.head(2)


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[25]:


data.shape


# In[26]:


X_train.shape


# In[27]:


X_test.shape


# In[28]:


#standardscalar
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[29]:


X_train


# 
# # Machine Learning Models

# # 
# Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)


# In[31]:


Y_pred = lr.predict(X_test)


# In[32]:


Y_pred


# In[33]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))    


# In[34]:


lr_acc = accuracy_score(Y_test,Y_pred)
print(lr_acc)


# In[35]:


results = pd.DataFrame()
results


# In[36]:


tempResults = pd.DataFrame({'Algorithm':['Logistics Regression Method'],'Accuracy':[lr_acc]})
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results


# # Decision Tree Classifier

# In[37]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)


# In[38]:


Y_pred = dtc.predict(X_test)


# In[39]:


Y_pred


# In[40]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))


# In[41]:


dtc_acc =(accuracy_score(Y_test,Y_pred))
dtc_acc


# In[42]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree Method'],'Accuracy':[dtc_acc]})
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results


# # Random Forest classifier

# In[43]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)


# In[44]:


Y_pred = rfc.predict(X_test)


# In[45]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_pred,Y_test)


# In[46]:


rfc_acc = accuracy_score(Y_pred,Y_test)
rfc_acc


# In[47]:


tempResults = pd.DataFrame({'Algorithm':['Random forest Method'],'Accuracy':[rfc_acc]})
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results


# # Support Vector Classifier

# In[48]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,Y_train)


# In[49]:


Y_pred = svc.predict(X_test)


# In[50]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_pred,Y_test)


# In[51]:


svc_acc =accuracy_score(Y_pred,Y_test)
svc_acc


# In[52]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Method'],'Accuracy':[svc_acc]})
results = pd.concat([results,tempResults])
results = results[['Algorithm','Accuracy']]
results


# In[ ]:





# In[ ]:




