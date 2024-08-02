import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
vr_dataset = pd.read_csv('C:/Users/Pero/Documents/Python Scripts/Kaggle/data.csv')

# Check dataset with print first 5 rows
vr_dataset.head()

# First, let's begin with dropping 'UserID' column. It is useless.
vr_dataset = vr_dataset.drop(columns=['UserID'])
# See if it worked
print(vr_dataset.head())

le = LabelEncoder()

# Select categorical columns using it's type. They are referred to as 'object'.
categorical_cols = vr_dataset.select_dtypes(include=['object']).columns

# Use LE with a lambda function to apply encoding to all selected columns
vr_dataset[categorical_cols] = vr_dataset[categorical_cols].apply(lambda col: le.fit_transform(col))

X = vr_dataset.drop(columns=['ImmersionLevel'])

X.head()

#Normalisieren
scaler = StandardScaler()

# Apply it to all columns
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(X.head())

# Split the data into training and test sets (!)
y = vr_dataset['ImmersionLevel']

# Split the X and y values to 30% test - 70% train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

#Ausgabe
print(accuracy_logistic)


svm_model = SVC()
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

#Ausgabe
print("------------")
print(accuracy_svm)


### Decision Tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

y_pred_decision_tree = decision_tree_model.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

#Ausgabe
print("--------------")
print(accuracy_decision_tree)


# Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

X_test = [[974, 0.02, 1.4, 0.9,0.5]]

y_pred_random_forest = random_forest_model.predict(X_test)
#accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)



#Ausgabe
print("-------------")
print(y_pred_random_forest)

