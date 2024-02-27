# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:25:40 2024

@author: sahil
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Assuming 'data' is your pandas DataFrame with the categorical dataset
# and 'target_column' is the column containing the target variable

train = pd.read_csv("C:/DS2/SalaryData_Train.csv")
train.shape
X_train = train.iloc[:,0:13]
Y_train= train.iloc[:,-1]

test  = pd.read_csv("C:/DS2/SalaryData_Test.csv")
X_test = test.iloc[:,0:13]
Y_test = test.iloc[:,-1]
# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(target_column, axis=1), data[target_column], test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
