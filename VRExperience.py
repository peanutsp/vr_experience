# PYTHON ROUTINE aus Kaggle.com #

# Initialer Import aller nötigen Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy



# Defition of Data Set Path
vr_dataset = pd.read_csv("RealisticModelData.csv")

# Check dataset with print first 5 rows
print(vr_dataset.head())

# Teste alle Variablen
print(vr_dataset.isnull().sum())

# Lösche alle Einträge in der ersten Spalte
vr_dataset = vr_dataset.drop(columns=['UserID'])

# Encode categorical columns
# Call the constructor of LE.
le = LabelEncoder()

# Select categorical columns using it's type. They are referred to as 'object'.
categorical_cols = vr_dataset.select_dtypes(include=['object']).columns

# Use LE with a lambda function to apply encoding to all selected columns
vr_dataset[categorical_cols] = vr_dataset[categorical_cols].apply(lambda col: le.fit_transform(col))

# Split the data into features and target variable
# Drop the dependent variable column.
Data = vr_dataset.drop(columns=['ImmersionLevel'])

# Normalize the features
# Call the constructor
scaler = StandardScaler()

# Apply it to all columns
Data = pd.DataFrame(scaler.fit_transform(Data), columns=Data.columns)

# Split the data into training and test sets (!)
data = vr_dataset['ImmersionLevel']


# Split the X and y values to 30% test - 70% train
Data_train, Data_test, data_train, data_test = train_test_split(Data, data, test_size=0.3, random_state=42)

# Train and test Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(Data_train, data_train)

y_pred_logistic = logistic_model.predict(Data_test)
accuracy_logistic = accuracy_score(data_test, y_pred_logistic)

print(y_pred_logistic)
print('Genauigkeit: ', accuracy_logistic)

print(' ')
print(' ')

# Train and test SVM model
svm_model = SVC()
svm_model.fit(Data_train, data_train)

y_pred_svm = svm_model.predict(Data_test)
accuracy_svm = accuracy_score(data_test, y_pred_svm)

print(y_pred_svm)
print('Genauigkeit: ', accuracy_svm)

print(' ')
print(' ')

# Train and test Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(Data_train, data_train)

y_pred_decision_tree = decision_tree_model.predict(Data_test)
accuracy_decision_tree = accuracy_score(data_test, y_pred_decision_tree)

print(y_pred_decision_tree)
print('Genauigkeit: ', accuracy_decision_tree)

print(' ')
print(' ')

# Train and test Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(Data_train, data_train)

y_pred_random_forest = random_forest_model.predict(Data_test)
accuracy_random_forest = accuracy_score(data_test, y_pred_random_forest)

print(y_pred_decision_tree)
print('Genauigkeit: ', accuracy_decision_tree)

print(' ')
print(' ')

# --- Ab hier werden MotionSickness Levels berechnet --- #

print(' ')
print('Ab hier werden MotionSickness Levels berechnet')
print(' ')

Data = vr_dataset.drop(columns=['MotionSickness'])

# Normalize the features
# Call the constructor
scaler = StandardScaler()

# Apply it to all columns
Data = pd.DataFrame(scaler.fit_transform(Data), columns=Data.columns)

# Split the data into training and test sets (!)
data = vr_dataset['MotionSickness']

# Split the X and y values to 30% test - 70% train
Data_train, Data_test, data_train, data_test = train_test_split(Data, data, test_size=0.3, random_state=42)

# Train and test Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(Data_train, data_train)

y_pred_logistic = logistic_model.predict(Data_test)
accuracy_logistic = accuracy_score(data_test, y_pred_logistic)

print(y_pred_logistic)
print('Genauigkeit: ', accuracy_logistic)

print(' ')
print(' ')

# Train and test SVM model
svm_model = SVC()
svm_model.fit(Data_train, data_train)

y_pred_svm = svm_model.predict(Data_test)
accuracy_svm = accuracy_score(data_test, y_pred_svm)

print(y_pred_svm)
print('Genauigkeit: ', accuracy_svm)

print(' ')
print(' ')

# Train and test Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(Data_train, data_train)

y_pred_decision_tree = decision_tree_model.predict(Data_test)
accuracy_decision_tree = accuracy_score(data_test, y_pred_decision_tree)

print(y_pred_decision_tree)
print('Genauigkeit: ', accuracy_decision_tree)

print(' ')
print(' ')

# Train and test Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(Data_train, data_train)

y_pred_random_forest = random_forest_model.predict(Data_test)
accuracy_random_forest = accuracy_score(data_test, y_pred_random_forest)

print(y_pred_decision_tree)
print('Genauigkeit: ', accuracy_decision_tree)

print(' ')
print(' ')

# Store the individual accuracies in a dictionary
#individual_accuracies = {
#    'Logistic Regression': accuracy_logistic,
#    'SVM': accuracy_svm,
#    'Decision Tree': accuracy_decision_tree,
#    'Random Forest': accuracy_random_forest
#}

# Plotting the individual accuracies
# model_names_individual = list(individual_accuracies.keys())
# accuracy_values_individual = list(individual_accuracies.values())
#
# plt.figure(figsize=(10, 6))
# plt.bar(model_names_individual, accuracy_values_individual, color=['blue', 'green', 'red', 'purple'])
# plt.ylabel('Accuracy')
# plt.title('Comparison of Classification Model Accuracies (Individual)')
# plt.ylim(0, 1)
# for i, v in enumerate(accuracy_values_individual):
 #   plt.text(i, v + 0.01, " {:.3f}".format(v), ha='center', color='black')

#plt.show()

# Count the number of each immersion level (!)
# immersion_counts = vr_dataset['ImmersionLevel'].value_counts()

# Plot the counts of each immersion level
# plt.figure(figsize=(10, 6))
# sns.countplot(x='ImmersionLevel', data=vr_dataset, palette='viridis')
# plt.title('Counts of Each Immersion Level', fontsize=16)
# plt.xlabel('Immersion Level', fontsize=14)
# plt.ylabel('Count', fontsize=14)
# plt.show()

# Plot Age vs ImmersionLevel
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='ImmersionLevel', y='Age', data=vr_dataset)
# plt.title('Age vs Immersion Level')
# plt.show()

# Plot Gender vs ImmersionLevel
# plt.figure(figsize=(10, 6))
# sns.countplot(x='ImmersionLevel', hue='Gender', data=vr_dataset)
# plt.title('Gender Distribution across Immersion Levels')
# plt.show()

# Plot VRHeadset vs ImmersionLevel
# plt.figure(figsize=(12, 6))
# sns.countplot(x='ImmersionLevel', hue='VRHeadset', data=vr_dataset)
# plt.title('VR Headset Usage across Immersion Levels')
# plt.show()