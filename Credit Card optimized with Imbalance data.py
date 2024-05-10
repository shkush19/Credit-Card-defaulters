import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
import time

# Start time
start_time = time.time()

# Load dataset
actualds = pd.read_excel(r"S:\Naresh IT\9th May\Self Learning\Credit card internship projects\default of credit card clients.xlsx", skiprows=1)

# Drop ID column as it's not useful for prediction
actualds.drop(columns=['ID'], inplace=True)

# Separate features (X) and target variable (y)
X = actualds.iloc[:, 0:23].values

y = actualds.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define a function to train, predict, and evaluate a classifier
def train_predict_evaluate_classifier(classifier):
    # Train classifier
    classifier.fit(X_train, y_train)
    
    # Predict classes for test set
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy on test set
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return accuracy

# Initialize classifiers
logit_classifier = LogisticRegression()
knn_classifier = KNeighborsClassifier()
svc_classifier = SVC()

# Calculate bias and variance for each classifier
bias_logit = train_predict_evaluate_classifier(logit_classifier)
bias_knn = train_predict_evaluate_classifier(knn_classifier)
bias_svc = train_predict_evaluate_classifier(svc_classifier)

variance_logit = bias_logit
variance_knn = bias_knn
variance_svc = bias_svc

# Create DataFrame with bias, variance, and accuracy
bias_variance_accuracy = pd.DataFrame({
    'Classifier': ['Logistic Regression', 'KNN', 'SVC'],
    'Bias': [bias_logit, bias_knn, bias_svc],
    'Variance': [variance_logit, variance_knn, variance_svc]
})

# Combine predicted values with original dataset
predicted_values_df = pd.DataFrame({
    'Logistic Regression': logit_classifier.predict(X_test),
    'KNN': knn_classifier.predict(X_test),
    'SVC': svc_classifier.predict(X_test)
})
combined_data = pd.concat([actualds, predicted_values_df], axis=1)

# Save the combined data to a CSV file
desired_directory = r'S:\Naresh IT\9th May\Self Learning\Credit card internship projects'
file_name = 'Classifier_combined.xlsx'
file_path = os.path.join(desired_directory, file_name)
combined_data.to_excel(file_path, index=False)

print("Combined data saved to:", file_path)
print(bias_variance_accuracy)

# End time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed Time (in seconds):", elapsed_time)

########################################



import matplotlib.pyplot as plt

# Plotting the class distribution
class_counts = actualds['default payment next month'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('default payment next month')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Defaulter', 'Defaulter'], rotation=0)
plt.show()

# Checking for imbalance
if len(class_counts) > 1:
    imbalance_ratio = class_counts.min() / class_counts.max()*100
    print("Imbalance ratio:", imbalance_ratio)
else:
    print("Dataset has only one class.")
    
################################################################################

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert X_resampled to DataFrame
X_resampled_df = pd.DataFrame(X_resampled)

# Concatenate resampled data into a new DataFrame
balanced_df = pd.concat([X_resampled_df, pd.DataFrame(y_resampled, columns=["default payment next month"])], axis=1)

# Now balanced_df contains the balanced dataset
print(balanced_df)

#########################################################
# Plotting the class distribution
class_counts = balanced_df['default payment next month'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('default payment next month')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Defaulter', 'Defaulter'], rotation=0)
plt.show()

# Checking for imbalance
if len(class_counts) > 1:
    imbalance_ratio = class_counts.min() / class_counts.max()*100
    print("Imbalance ratio:", imbalance_ratio)
else:
    print("Dataset has only one class.")










