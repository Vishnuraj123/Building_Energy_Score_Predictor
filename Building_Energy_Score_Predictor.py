# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 03:09:22 2020

@author: vishnu.r
"""

# Simple Linear Regression

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

#Funtion for Various Use
def reject_outliers(dataframe, m):
    return dataframe[abs(dataframe - np.mean(dataframe)) < m * np.std(dataframe)]    

def norm(dataframe):
    dataframe= (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
    return dataframe
   
def dummy_coding(dataframe):
     categorical_variables = dataframe.dtypes.loc[dataframe.dtypes == 'object'].index
     data = dataframe[categorical_variables]
     dummies = pd.get_dummies(data)
     dataframe = pd.concat([dataframe, dummies], axis=1)
     dataframe = dataframe.drop(categorical_variables, axis=1)
     return dataframe


# Dataset Preparation
dataset = pd.read_csv('input.csv')
dataset=dataset[[ 'Property Id',  'Borough', 'DOF Gross Floor Area',
       'Primary Property Type - Self Selected',
        'Year Built', 'Number of Buildings - Self-reported', 'Occupancy',
       'Metered Areas (Energy)', 'Metered Areas  (Water)',
       'ENERGY STAR Score', 'Site EUI (kBtu/ft²)',
       'Weather Normalized Site EUI (kBtu/ft²)',
       'Weather Normalized Site Electricity Intensity (kWh/ft²)',
       'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
       'Weather Normalized Source EUI (kBtu/ft²)',
       'Natural Gas Use (kBtu)',
       'Weather Normalized Site Natural Gas Use (therms)',
       'Electricity Use - Grid Purchase (kBtu)',
       'Weather Normalized Site Electricity (kWh)',
       'Total GHG Emissions (Metric Tons CO2e)',
       'Direct GHG Emissions (Metric Tons CO2e)',
       'Indirect GHG Emissions (Metric Tons CO2e)',
       'Property GFA - Self-Reported (ft²)',
       'Water Use (All Water Sources) (kgal)',
       'Water Intensity (All Water Sources) (gal/ft²)',
       'Source EUI (kBtu/ft²)', 'Release Date', 'Water Required?',
       'DOF Benchmarking Submission Status', 'Latitude', 'Longitude']]

#rename columns
dataset.columns=['Property Id', 'Borough', 'DOF Gross Floor Area',
       'Primary Property Type - Self Selected', 'Year Built',
       'Number of Buildings - Self-reported', 'Occupancy',
       'Metered Areas (Energy)', 'Metered Areas  (Water)',
       'ENERGY STAR Score','Site EUI',
       'Weather Normalized Site EUI',
       'Weather Normalized Site Electricity Intensity',
       'Weather Normalized Site Natural Gas Intensity',
       'Weather Normalized Source EUI',
       'Natural Gas Use',
       'Weather Normalized Site Natural Gas Use',
       'Electricity Use - Grid Purchase',
       'Weather Normalized Site Electricity',
       'Total GHG Emissions',
       'Direct GHG Emissions',
       'Indirect GHG Emissions',
       'Property GFA - Self-Reported',
       'Water Use (All Water Sources)',
       'Water Intensity (All Water Sources)',
       'Source EUI', 'Release Date', 'Water Required',
       'DOF Benchmarking Submission Status', 'Latitude', 'Longitude']
#Missing Value Treatment
features_drop=['Borough', 'DOF Gross Floor Area','Primary Property Type - Self Selected',
               'Year Built',
               'Number of Buildings - Self-reported',
               'Occupancy',
               'Metered Areas (Energy)',
               'ENERGY STAR Score',
               'Site EUI',
               'Source EUI',
               'Release Date',
               'Water Required',
               'DOF Benchmarking Submission Status']

features_ave=['Weather Normalized Site Electricity Intensity'
              'Weather Normalized Site Natural Gas Intensity'
              'Weather Normalized Source EUI'
              'Natural Gas Use'
              'Weather Normalized Site Natural Gas Use'
              'Electricity Use - Grid Purchase'
              'Weather Normalized Site Electricity'
              'Total GHG Emissions'
              'Direct GHG Emissions'
              'Indirect GHG Emissions'
              'Latitude'
              'Longitude'
              ]

for col in features_drop:
    dataset=dataset[dataset[col].notnull()]
    dataset=dataset[dataset[col]!="Not Available"]
    
for col in features_ave:
    dataset[col]=dataset[col].fillna(dataset[col].mean())


#Outlier Treatment
features_OT=['Year Built',
'Number of Buildings - Self-reported',
'Occupancy',
'ENERGY STAR Score',
'Site EUI',
'Weather Normalized Site Electricity Intensity',
'Weather Normalized Site Natural Gas Intensity',
'Natural Gas Use',
'Total GHG Emissions',
'Property GFA - Self-Reported',
'Latitude',
'Longitude'
]

dataset=reject_outliers(dataset)

#Dummy Coding and Normalization
dataset=dummy_coding(dataset)   
dataset[col]=norm(dataset[col])

train_columns=dataset.columns.values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(dataset[train_columns], dataset['ENERGY STAR Score'], test_size = .25, random_state = 0)

# Training the Multiple ML model on the Training set
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

regressor2 = DecisionTreeRegressor()
regressor2.fit(X_train, y_train)

regressor3 = GradientBoostingRegressor()
regressor3.fit(X_train, y_train)

regressor4 = RandomForestRegressor()
regressor4.fit(X_train, y_train)

print("Linear Regressor 1-RSquare:",r2_score(y_test,regressor1.predict(X_test)))
print("Decision Tree Regressor 2-RSquare:",r2_score(y_test,regressor2.predict(X_test)))
print("Gradient Boosting Regressor 3-RSquare:",r2_score(y_test,regressor3.predict(X_test)))
print("Random Forest Regressor 4-RSquare:",r2_score(y_test,regressor4.predict(X_test)))

print("GBM Regressor 1-RSquare:",r2_score(y_train,regressor1.predict(X_train)))
print("GBM Regressor 2-RSquare:",r2_score(y_train,regressor2.predict(X_train)))
print("GBM Regressor 3-RSquare:",r2_score(y_train,regressor3.predict(X_train)))
print("GBM Regressor 4-RSquare:",r2_score(y_train,regressor4.predict(X_train)))

#Hyper parameter tuning of GBM

regressor1 = GradientBoostingRegressor(n_estimators=50,max_depth=4)
regressor1.fit(X_train, y_train)

regressor2 = GradientBoostingRegressor(n_estimators=100,max_depth=3)
regressor2.fit(X_train, y_train)

regressor3 = GradientBoostingRegressor(n_estimators=150,max_depth=6)
regressor3.fit(X_train, y_train)

regressor4 = GradientBoostingRegressor(n_estimators=50,max_depth=6)
regressor4.fit(X_train, y_train)

print("GBM Regressor 1 1-RSquare:",r2_score(y_test,regressor1.predict(X_test)))
print("GBM Regressor 2-RSquare:",r2_score(y_test,regressor2.predict(X_test)))
print("GBM Regressor 3-RSquare:",r2_score(y_test,regressor3.predict(X_test)))
print("GBM Regressor 4-RSquare:",r2_score(y_test,regressor4.predict(X_test)))

df = pd.DataFrame()
df['Columns'] = train_columns
df['importance'] =regressor1.feature_importances_

df.to_csv("FI.csv")

