#!/usr/bin/env python
# coding: utf-8

# In[39]:


import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics



# In[40]:


pwd


# In[41]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='YrQegtEAFTtgFLwpCLbzHKmafzrTqvXux9ly58CLUtL4',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'flightdelayprediction-donotdelete-pr-6pklnqsuhe0vn6'
object_key = 'flightdata.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)



# In[42]:


dataset.head()


# In[43]:


dataset.info()


# In[44]:


dataset.describe()


# In[45]:


dataset.isnull().sum()


# In[46]:


dataset["DEST"].unique()


# In[47]:


sns.scatterplot(x='ARR_DELAY',y='ARR_DEL15',data=dataset)


# In[48]:


sns.catplot(x="ARR_DEL15",y="ARR_DELAY",kind='bar',data=dataset)


# In[49]:


sns.heatmap(dataset.corr())


# In[50]:


dataset=dataset.drop('Unnamed: 25',axis=1)
dataset.isnull().sum()


# In[51]:


dataset =dataset[["FL_NUM","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","ORIGIN","DEST","CRS_ARR_TIME","DEP_DEL15","ARR_DEL15"]]
dataset.isnull().sum()


# In[52]:


dataset=dataset.fillna({'ARR_DEL15':1})
dataset=dataset.fillna({'DEP_DEL15':0})
dataset.iloc[117:185]


# In[53]:


import math 

for index,row in dataset.iterrows():
        dataset.loc[index,'CRS_ARR_TIME'] = math.floor(row['CRS_ARR_TIME']/100)
dataset.head()


# In[54]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['DEST'] = le.fit_transform(dataset['DEST'])
dataset['ORIGIN'] = le.fit_transform(dataset['ORIGIN'])


# In[55]:


dataset.head(5)


# In[56]:


from sklearn.preprocessing import OneHotEncoder
x=dataset.values
oh = OneHotEncoder()
z=oh.fit_transform(x[:,4:5]).toarray()
t=oh.fit_transform(x[:,5:6]).toarray()


# In[57]:


z


# In[58]:


t


# In[59]:


dataset = pd.get_dummies(dataset,columns=['ORIGIN','DEST'])
dataset.head()


# In[60]:


x=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8:9].values


# In[61]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[62]:


x_test.shape


# In[63]:


y_train.shape


# In[64]:


x_train.shape


# In[65]:


y_test.shape


# In[66]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[67]:


from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(random_state = 0)
classifier.fit(x_train,y_train)


# In[68]:


decisiontree = classifier.predict(x_test)


# In[69]:


decisiontree


# In[70]:


from sklearn.metrics import accuracy_score
desacc = accuracy_score (y_test,decisiontree)


# In[71]:


desacc


# In[72]:


import pickle


# In[73]:


pickle.dump(classifier,open('flight.pkl','wb'))


# In[74]:


pip install ibm_watson_machine_learning


# In[75]:


from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"SIPG1xOrBleCVP2trdmcnOuwL2SD_fe9i5doB2akj1gb"
}
client = APIClient(wml_credentials)


# In[76]:


def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    return (next(item for item in space['resources'] if item['entity']["name"]==spacename)['metadata']['id'])


# In[ ]:


space_uid = guid_from_space_name(client,'model')
print("Space UID = "+space_uid)


# In[ ]:


client.set.default_space(space_uid)


# In[ ]:




