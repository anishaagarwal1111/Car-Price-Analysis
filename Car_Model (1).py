#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv("CarData.csv")


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.carlength.value_counts()


# In[8]:


#visualisation
#Make bins of price and count it
ax = df.plot.bar(x='carlength', y='price', rot=0)
plt.show()


# In[9]:


#make bins using replace
plt.figure();
df["price"].diff().hist();


# In[10]:


# to check  categorical variables
for i in df.columns:  
 print(i,df[i].nunique())


# In[11]:


df.drop(['car_ID','CarName'], axis=1, inplace=True)


# In[12]:


df.fueltype.unique()


# In[13]:


df.replace({"fueltype":{"gas":0,"diesel":1}},inplace=True)


# In[14]:


df.aspiration.unique()


# In[15]:


df.replace({"aspiration":{"std":0,"turbo":1}},inplace=True)


# In[16]:


df.replace({"doornumber":{"two":0,"four":1}},inplace=True)


# In[17]:


df.replace({"enginelocation":{"front":0,"rear":1}},inplace=True)


# In[18]:


df=pd.get_dummies(df,columns=["carbody","drivewheel","enginetype","fuelsystem","cylindernumber"],drop_first=True)


# In[19]:


df.head(4)


# In[20]:


X=df.drop(["price"],axis=1)
y=df["price"]


# In[21]:


#normalisation

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X[X.columns] = scaler.fit_transform(X)
X.head(5)


# #### Now check the assumptions for linear regression: should be linear indepenent,should'nt be multicollinear, should follow homoscedasticity

# In[22]:


import seaborn as sns


# In[23]:


# p=sns.pairplot(df)


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


corrMatrix=df.corr()
f,ax=plt.subplots(figsize=(11,9))
sns.heatmap(corrMatrix,annot=True)
plt.show()


# In[26]:


from statsmodels.stats.outliers_influence import variance_inflation_factor  ##removing multicollinear variables


# In[27]:


def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]


# In[28]:


calculate_vif_(X)


# In[29]:


X=df[['aspiration', 'doornumber', 'enginelocation', 'peakrpm',
       'carbody_hardtop', 'carbody_hatchback', 'carbody_wagon',
       'drivewheel_rwd', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohcf',
       'enginetype_ohcv', 'fuelsystem_2bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi', 'fuelsystem_spdi',
       'fuelsystem_spfi', 'cylindernumber_five', 'cylindernumber_six',
       'cylindernumber_three', 'cylindernumber_twelve']]
y=df['price']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=42)


# In[32]:


X


# In[33]:


X.shape


# In[34]:


y.shape


# In[35]:


import statsmodels.api as sm 


# In[36]:


lm = sm.OLS(y_train,X_train).fit()


# In[37]:


lm.summary()


# In[38]:


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

def homoscedasticity_assumption(model, features, label):
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=1)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(True) 
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show() 


# In[39]:


homoscedasticity_assumption(lm, X_train, y_train)


# In[40]:


lm.predict(X_test)


# In[ ]:





# In[ ]:




