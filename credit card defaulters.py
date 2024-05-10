'''##############################################Predict Credit Cards Defaulters#######################################################################'''

'''Describing Data set attributes'''
'''
Column 0(X0):---ID: Unique identifier for each individual or account.
Column 1(X1):---LIMIT_BAL: Credit limit for the account.
Column 2(X2):---SEX: Gender of the individual (1 = male, 2 = female).
Column 3(X3):---EDUCATION: Level of education (1 = graduate school, 2 = university, 3 = high school, 4 = others).
Column 4(X4):---MARRIAGE: Marital status (1 = married, 2 = single, 3 = others).
Column 5(X5):---AGE: Age of the individual.
Column 6 to 11(X6 to X11):---PAY_0 to PAY_6: Repayment status in the last 6 months. 
                  Negative numbers indicate the number of months of delay in payment, 
                  while positive numbers indicate the number of months paid in advance. 
                  For example-- '-1' indicates payment delay for one month, 
                                '-2' indicates payment delay for two months, 
                                '0' indicates the payment was made on time.
Column 12 to 17(X12 to X17):---BILL_AMT1 to BILL_AMT6: Bill amount (in dollars) for the last 6 months.
Column 18 to 23(X18 to X23):---PAY_AMT1 to PAY_AMT6: Amount of payment made (in dollars) for the last 6 months.
Column 24(X24):-default payment next month: Binary variable indicating whether the individual defaulted on the credit card payment next month (1 = yes, 0 = no).'''

'''Aim is to predict the Realtime defaulters of credit card by applying different models and suggest the best model basis on accuracy percentages'''


'''Let's start with importing libraries'''
import time

# Start time
start_time = time.time()

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os

# Load dataset
actualds = pd.read_excel(r"S:\Naresh IT\9th May\Self Learning\Credit card internship projects\default of credit card clients.xlsx", skiprows=1)

# Let's make a copy before dropping any irrelevant column.
actualds1 = actualds.copy()

# Drop ID column as it's not useful for prediction
actualds1.drop(columns=['ID'], inplace=True)

# Separate features (X) and target variable (y)
X = actualds1.iloc[:, 0:23].values
y = actualds1.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize classifiers
logit_classifier = LogisticRegression()
knn_classifier = KNeighborsClassifier()
svc_classifier = SVC()

# Train classifiers
logit_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
svc_classifier.fit(X_train, y_train)

# Predict classes for test set
y_pred_logit = logit_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_svc = svc_classifier.predict(X_test)

# Calculate bias (accuracy on training set)
bias_logit = logit_classifier.score(X_train, y_train)
bias_knn = knn_classifier.score(X_train, y_train)
bias_svc = svc_classifier.score(X_train, y_train)

# Calculate variance (accuracy on test set)
variance_logit = logit_classifier.score(X_test, y_test)
variance_knn = knn_classifier.score(X_test, y_test)
variance_svc = svc_classifier.score(X_test, y_test)

# Calculate accuracy scores in percentages
ac_logit = accuracy_score(y_test, y_pred_logit)*100
ac_knn = accuracy_score(y_test, y_pred_knn)*100
ac_svc = accuracy_score(y_test, y_pred_svc)*100


print("Logistic Regression Bias:", bias_logit)
print("Logistic Regression Variance:", variance_logit)
print("KNN Classifier Bias:", bias_knn)
print("KNN Classifier Variance:", variance_knn)
print("SVC Bias:", bias_svc)
print("SVC Variance:", variance_svc)
print("Logistic Regression Accuracy:", ac_logit)
print("KNN Classifier Accuracy:", ac_knn)
print("SVC Accuracy:", ac_svc)

predicted_values_df = pd.DataFrame({
    'Logistic Regression': logit_classifier.predict(X_test),
    'KNN': knn_classifier.predict(X_test),
    'SVC': svc_classifier.predict(X_test)
  
})

# Concatenate the predicted values with the original dataset
combined_data = pd.concat([actualds, predicted_values_df], axis=1)

# Save the combined data to a CSV file
combined_data.to_excel('Classifier_combined.xlsx', index=False)


# Desired directory path
desired_directory = r'S:\Naresh IT\9th May\Self Learning\Credit card internship projects'

# Change the current working directory
os.chdir(desired_directory)

# Now, save the file in the desired location
file_name = 'Classifier_combined.xlsx'
file_path = os.path.join(r"S:\Naresh IT\9th May\Self Learning\Credit card internship projects", file_name)

# Assuming you have some DataFrame called 'data' to save
combined_data.to_excel(file_path, index=False)

# Calculate bias and variance for Logistic Regression and KNN
bias_variance_accuracy = {
    'Classifier': ['Logistic Regression', 'KNN', 'SVC'],
    'Bias': [bias_logit, bias_knn, bias_svc],
    'Variance': [variance_logit, variance_knn, variance_svc],
    'Accuracy': [ac_logit, ac_knn, ac_svc]
}


# Create DataFrame
bias_variance_accuracy = pd.DataFrame(bias_variance_accuracy)

print(bias_variance_accuracy)

# End time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed Time (in seconds):", elapsed_time)








