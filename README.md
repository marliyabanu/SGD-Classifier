# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
 1. Import the required libraries and load the Iris dataset.

2. Split the dataset into training data and testing data.

3. Apply feature scaling using StandardScaler.

4. Create and train the SGD Classifier model using the training data.

5. Predict the test data results and evaluate the model using accuracy score, classification report, and confusion matrix.
```

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MARLIYA BANU B 
RegisterNumber: 212225040229
*/
# SGD Classifier using Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Classifier Model
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```


## Output:
<img width="587" height="386" alt="WhatsApp Image 2026-05-11 at 9 45 18 AM" src="https://github.com/user-attachments/assets/97f166b7-90a9-4ef1-a042-7a26f4234c0d" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
