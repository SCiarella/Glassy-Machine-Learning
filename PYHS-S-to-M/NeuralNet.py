import pandas as pd
import glob
import seaborn as sns
import math
import sys
import matplotlib.pyplot as plt
import random

# leave to -1 to read all the data in the folder
NS_to_open=-1

# The input data consists of the following informations
input_labels=[]
for si in range(100):
    input_labels.append('S%d'%si)
input_labels.append('eta')
# and those are the output 
input_labels.append('log t')
input_labels.append('F')

full_df=pd.DataFrame()


print('\n*******************\n\t Creating input database: ')
listSfiles= glob.glob('Data/Sk_data/Sk_eta*')
for Sfile in listSfiles[:NS_to_open]:
    # The first 100 + 1 columns are for the input data
    print('Reading %s'%Sfile)
    eta=(Sfile.split('_eta_')[1]).split('.txt')[0]
    new_in_data=(pd.read_csv(Sfile, header=None, sep='\n')).transpose()
    new_in_data=pd.concat([new_in_data,pd.DataFrame([eta])],axis=1)

#    print(new_in_data)

    # then for this input we have all the time dependent output that I store as additional columns 
    phifile='Data/phi_data/phi_eta_'+eta+'.txt'
    print('Reading %s'%phifile)
    new_out_data=pd.read_csv(phifile, header=None, sep='\t')
    for index,line in new_out_data.iterrows():
        new_line=pd.concat([new_in_data,pd.DataFrame([line[0],line[1]]).transpose()],axis=1)
#        print(new_line)

        # Add the line that contains IN + OUT to the full database
        full_df = pd.concat([full_df,new_line])
#        print(full_df)

full_df.columns=input_labels
full_df=full_df.reset_index(drop=True)
print(full_df)

# Here we can shuffle the database as we want
#   [ ... ]

# Separate the data in X,Y
X = full_df.drop(['F'], axis=1)
print(X.head())
print(X.shape)


y = full_df[['F']]
print(y.head())
print(y.shape)



# Use sklearn to make train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)



# Here we can normalize the data or do any pre-processing
import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()

feature_names = X.columns[1:]
print(feature_names)
output_names = y.columns[:]
print(output_names)

X_train_scaled = X_train.copy()
X_train_scaled[feature_names] = min_max_scaler.fit_transform(X_train[feature_names])
X_train_scaled.describe()


X_test_scaled = X_test.copy()
X_test_scaled[feature_names] = min_max_scaler.fit_transform(X_test[feature_names])

y_test_scaled = y_test.copy()
y_test_scaled[output_names] = min_max_scaler.fit_transform(y_test[output_names])

y_train_scaled = y_train.copy()
y_train_scaled[output_names] = min_max_scaler.fit_transform(y_train[output_names])





from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,100), activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
                     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, 
                     shuffle=True, random_state=666, tol=0.0001, verbose=True, warm_start=False, 
                     momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                     beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=150000)




# Here I do bayesian (or grid search) to determine the best parameters for the model
# (unfortunately bayesian search does not work for now)

list_of_architectures = [(50, 50, 50), (50, 100, 50), (100,),
                           (100, 100), (100, 100, 100), (50,)]


##from skopt import BayesSearchCV
### The parameter space is a dictionary  
##parameter_space = dict()
##parameter_space['alpha'] = (1e-6, 1e-2, 'log-uniform')
##parameter_space['hidden_layer_sizes'] = [(10,10),(100)] 
##parameter_space['n_iter_no_change'] = (10,500, 'log-uniform')
##parameter_space['solver'] = ['sgd','adam']
##parameter_space['tol'] = (1e-5,1e-3, 'log-uniform')
##from sklearn.model_selection import RepeatedStratifiedKFold
##cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
##search = BayesSearchCV(model, parameter_space, n_jobs=-1, cv=cv)

# This is gridSearch, if Bayes is not ok
from sklearn.model_selection import GridSearchCV
parameter_space = {
    'alpha': [0.0001, 0.001, 0.05],
    'hidden_layer_sizes':  list_of_architectures,
    'n_iter_no_change': [10, 100, 1000],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'tol': [10],
    #'tol': [10**-3,10 ** -4, 10 ** -5],
}
search = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3)

search.fit(X_train_scaled,y_train_scaled)

# Best parameter set
print('Best parameters found:\n', search.best_params_)


print('R2 score over the training set: %f'%search.score(X_train_scaled, y_train_scaled))
print('R2 score over the validation set: %f'%search.score(X_test_scaled, y_test_scaled))
