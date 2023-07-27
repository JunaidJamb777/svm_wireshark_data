import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data_df = pd.read_csv('cleaned_data.csv')

# Function to extract URLs from the "Info" column
def extract_url(info):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?://\S+)'
    urls = re.findall(url_pattern, info)
    return urls

# Apply the extract_url function to the "Info" column and create a new column with extracted URLs
data_df['URLs'] = data_df['Info'].apply(extract_url)

# Create a new DataFrame containing only the relevant columns
new_data_df = data_df[['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'URLs', 'Info']]

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data = encoder.fit_transform(new_data_df[['Source', 'Destination', 'Protocol']])

# Save the encoder to a file
joblib.dump(encoder, 'encoder.pkl')

# Split the data into features (X) and target variable (y)
X = encoded_data
y = new_data_df['Info']
print(X,y)
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Save the trained SVM classifier to a file
joblib.dump(svm_classifier, 'svm_classifier.pkl')
print("done svm")
# Load the trained SVM classifier and encoder from the files
loaded_encoder = joblib.load('encoder.pkl')
loaded_svm_classifier = joblib.load('svm_classifier.pkl')

# Transform new data using the loaded encoder
new_data = pd.DataFrame({'Source': ['source_value'], 'Destination': ['destination_value'], 'Protocol': ['protocol_value']})
encoded_new_data = loaded_encoder.transform(new_data)
print("done model")
# Make predictions using the loaded SVM classifier
predictions = loaded_svm_classifier.predict(encoded_new_data)
print("Predictions:", predictions)

# Make predictions on the testing data
y_pred = loaded_svm_classifier.predict(X_test)

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
