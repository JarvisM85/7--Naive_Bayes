import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data
SalaryData_Train = pd.read_csv("C:/DS2/SalaryData_Train.csv")

# Display first few rows of the dataset
print(SalaryData_Train.head())

# Identify features (X) and target variable (y)
X = SalaryData_Train.drop('Salary', axis=1)  # Features
y = SalaryData_Train['Salary']  # Target variable

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')