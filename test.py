import pandas as pd
import joblib

# Load the test dataset
test_data = pd.read_csv('cleaned_data.csv')

# Load the trained SVM classifier and encoder
svm_classifier = joblib.load('svm_classifier.pkl')
encoder = joblib.load('encoder.pkl')

# Encode the categorical variables in the test data
categorical_columns = ['Source', 'Destination', 'Protocol']
for column in categorical_columns:
    test_data[column] = encoder.transform(test_data[column])

# Make predictions using the SVM model
predictions = svm_classifier.predict(test_data.drop("Info", axis=1))

# Print the predictions
print(predictions)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
