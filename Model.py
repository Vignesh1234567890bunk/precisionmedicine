import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv('healthcare_dataset.csv')
print(df.head())

print(df.shape)

print(df.isna().sum())


plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()

# Boxplot of Billing Amount by Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
plt.xlabel('Admission Type')
plt.ylabel('Billing Amount')
plt.title('Billing Amount by Admission Type')
plt.show()


for col in df.columns:
    if col!='Age':
        print(col, df[col].value_counts())

print(df.columns)

# feature Engineering
# not needed columns - [Name, Date, Doctor, Hospital, Insurance, amount, room, discharge]
df = df[[ 'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Medication', 'Test Results']]
print(df.head())



from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
for col in df.columns:
    if col!='Age':
        df[col]=lc.fit_transform(df[col])
        for i, category in enumerate(lc.classes_):
            print(f"{category} is mapped to {i}")
print(df.head())



X,y=df.drop(['Test Results'],axis=1), df['Test Results']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)
print(y_test.shape)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

filename = 'rfc-model.pkl'
pickle.dump(rfc, open(filename, 'wb'))
rfc.score(X_test,y_test)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))