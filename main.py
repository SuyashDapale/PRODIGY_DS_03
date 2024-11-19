import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

url = "C:/Users/Suyash Dapale/Desktop/Prodigy Infotech Internship/PRODIGY_DATA-SCIENCE_TASKS/PRODIGY_DS_03/bank.csv"
data = pd.read_csv(url, sep=';')

# Display basic information
print(data.head())


# Encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop(columns=['y'])  # y is the target column
y = data['y']  # 'y' represents whether a customer subscribed or not (encoded as 0/1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Instantiate the classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)

# Train the model
clf = clf.fit(X_train, y_train)


# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))



plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True)
plt.show()
