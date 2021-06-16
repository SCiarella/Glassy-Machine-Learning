#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import seaborn as sns


# The input consists in 60 features and we have a total of 355 entries

# In[14]:


X = pd.read_csv('x.csv')
print(X.head())
print(X.shape)


# We want to predict 27 floating point outputs from the input data

# In[15]:


y = pd.read_csv('y.csv')
print(y.head())
print(y.shape)


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[17]:


import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()


# In[18]:


feature_names = X.columns[1:]
print(feature_names)
output_names = y.columns[1:]
print(output_names)


# In[19]:


X_train_scaled = X_train.copy()
X_train_scaled[feature_names] = min_max_scaler.fit_transform(X_train[feature_names])
X_train_scaled.describe()


# In[20]:


X_test_scaled = X_test.copy()
X_test_scaled[feature_names] = min_max_scaler.fit_transform(X_test[feature_names])

y_test_scaled = y_test.copy()
y_test_scaled[output_names] = min_max_scaler.fit_transform(y_test[output_names])

y_train_scaled = y_train.copy()
y_train_scaled[output_names] = min_max_scaler.fit_transform(y_train[output_names])


# In[21]:


from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,100), activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
                     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, 
                     shuffle=True, random_state=666, tol=0.0001, verbose=True, warm_start=False, 
                     momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                     beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=150000)





# In[ ]:


from sklearn.model_selection import GridSearchCV
parameter_space = {
    #'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.05],
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,),
                           (100, 100), (100, 100, 100), (50,)],
    'n_iter_no_change': [10, 100, 1000],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'tol': [10**-3,10 ** -4, 10 ** -5],
}
print("Creating GridSearchCV object")
model = GridSearchCV(model, parameter_space, n_jobs=4, cv=3)
print("Starting Grid Search (WARNING: this may take a VERY long time)")
model.fit(X_train_scaled,y_train_scaled)
print("Finished Grid Search")

# Best parameter set
print('Best parameters found:\n', clf.best_params_)


# In[ ]:


#model.fit(X_train_scaled,y_train_scaled)


# R2 score over the train set

# In[ ]:


model.score(X_train_scaled, y_train_scaled)


# R2 score over test set

# In[ ]:


model.score(X_test_scaled, y_test_scaled)


# In[ ]:





# In[ ]:





# In[ ]:




