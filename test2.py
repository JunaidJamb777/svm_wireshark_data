import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the test dataset
test_data = pd.read_csv('cleaned_data.csv')

# Load the trained SVM classifier and encoder
svm_classifier = joblib.load('svm_classifier.pkl')
# encoder = joblib.load('encoder.pkl')

# # Encode the categorical variables in the test data
# categorical_columns = ['source', 'destination', 'protocol']
# for column in categorical_columns:
#     test_data[column] = encoder.transform(test_data[column])


# Load the encoder from the file
loaded_encoder = joblib.load('encoder.pkl')

# Transform new data using the loaded encoder
new_data = pd.DataFrame({'Source': ['source_value'], 'Destination': ['destination_value'], 'Protocol': ['protocol_value']})
encoded_new_data = loaded_encoder.transform(new_data)
test_data = new_data


# Extract the features (X) and target variable (y) from the test data
X_test = test_data.drop("Info", axis=1)
y_test = test_data["Info"]

# Make predictions using the SVM model
predictions = svm_classifier.predict(X_test)

# Print the predictions
print(predictions)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
