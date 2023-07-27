import pandas as pd
data_df=pd.read_csv("27_7_2023_10_57.csv")
# Check for missing values
print(data_df.isnull().sum())

# Handle missing values by either removing rows with missing values or imputing them
data_df = data_df[~data_df['Source'].str.contains(r'cisco', case=False)]

# Option 1: Remove rows with missing values
data_df = data_df.dropna()

# Option 2: Impute missing values with mean or median
# For example, if you want to impute missing values in the 'Length' column with the mean value:
mean_length = data_df['Length'].mean()
data_df['Length'] = data_df['Length'].fillna(mean_length)

import numpy as np

# Calculate Z-Scores for each numerical column
z_scores = np.abs((data_df['Length'] - data_df['Length'].mean()) / data_df['Length'].std())

# Set a threshold for Z-Score (e.g., 3) to identify outliers
threshold = 3

# Filter out rows with outliers
data_df = data_df[z_scores < threshold]

# Reset the index of the DataFrame
data_df.reset_index(drop=True, inplace=True)

import re

# Function to extract URLs from the "Info" column
def extract_url(info):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?://\S+)'
    urls = re.findall(url_pattern, info)
    return urls

# Apply the extract_url function to the "Info" column and create a new column with extracted URLs
data_df['URLs'] = data_df['Info'].apply(extract_url)

# Create a new DataFrame containing all the columns, including the "Info" and "URLs" columns
new_data_df = data_df[['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info', 'URLs']]

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data_df = encoder.fit_transform(new_data_df[['Source', 'Destination', 'Protocol']])

# Save the encoder to a file
joblib.dump(encoder, 'encoder.pkl')

# Load the encoder from the file
loaded_encoder = joblib.load('encoder.pkl')

# Transform new data using the loaded encoder
new_data = pd.DataFrame({'Source': ['source_value'], 'Destination': ['destination_value'], 'Protocol': ['protocol_value']})
encoded_new_data = loaded_encoder.transform(new_data)
svm_classifier = joblib.load('svm_classifier.pkl')
predictions = svm_classifier.predict(encoded_new_data)



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
