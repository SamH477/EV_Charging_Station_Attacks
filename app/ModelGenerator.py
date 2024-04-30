import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

df = pd.read_csv('Power_Consumption.csv')

minority_class = df[df['Label'] == 'benign']
majority_class = df[df['Label'] == 'attack']


minority_upsampled = resample(minority_class, replace=True, n_samples=26726, random_state=101)
majority_downsampled = resample(majority_class, replace=False, n_samples=26726, random_state=101)

# Combine the upsampled minority class with the majority class
df = pd.concat([majority_downsampled, minority_upsampled])

cs = pd.get_dummies(df['State'], drop_first = False)
df = pd.concat([df, cs], axis = 1)

def update_labels(cols):
    outcome = cols[0]
    if outcome == 'attack':
        return 1
    else:
        return 0
    
df['Label'] = df[['Label']].apply(update_labels, axis = 1)

df.drop(['interface', 'Attack', 'idle', 'State', 'Attack-Group'], axis = 1, inplace = True)

scaler = StandardScaler()
X = df.drop(['time', 'Label'], axis = 1)
y = df['Label']

X_scaled = scaler.fit_transform(X)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

svcmodel = SVC(C = 10, kernel = 'rbf', gamma = 1, random_state = 101, probability= True)
svcmodel.fit(X_train, y_train)
with open('svc_model.pkl', 'wb') as file:
    pickle.dump(svcmodel, file)

#code to open file
#with open('svc_model.pkl', 'rb') as file:
#    loaded_model = pickle.load(file)