

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

"""# Reading the dataset"""

df=pd.read_csv('employee-attrition.csv')


df['Attrition'].replace({'Yes': 1, 'No': 0}, inplace=True)

#data with attrition =yes
data_with_attrition_yes = df[df['Attrition'] == 1]
data_with_attrition_yes.sample(10)



# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Fill null values for numerical columns using a for loop
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill null values for categorical columns using a for loop
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.isna().sum()


col=['NumCompaniesWorked','MonthlyIncome','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
q1=df[col].quantile(0.25)
q3=df[col].quantile(0.75)
iqr=q3-q1
iqr

lw_bd=q1-1.5*iqr
up_bd=q3+1.5*iqr
lower=df[col]<lw_bd
upper=df[col]>up_bd
df[col]=df[col].clip(lower=lw_bd,upper=up_bd,axis=1)


df.drop('HourlyRate', axis=1, inplace=True)
df.drop('DailyRate', axis=1, inplace=True)

df['Over18'].value_counts()

df.drop('Over18', axis=1, inplace=True)



df['BusinessTravel'].value_counts()

df['BusinessTravel'].replace({'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}, inplace=True)

df['Gender'].replace({'Female': 0, 'Male': 1}, inplace=True)


df['Department'].value_counts()

df['Department'].replace({'Research & Development': 0, 'Sales': 1, 'Human Resources': 2}, inplace=True)

df['JobRole'].value_counts()

df['MaritalStatus'].value_counts()

df['MaritalStatus'].replace({'Single': 0, 'Married': 1, 'Divorced': 2}, inplace=True)

df['EducationField'].value_counts()

df['EducationField'].replace({'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5}, inplace=True)
df['OverTime'].replace({'No': 0, 'Yes': 1}, inplace=True)



"""# selecting features with the help of selectKbest"""

from sklearn.feature_selection import SelectKBest,chi2, f_classif

from sklearn.feature_selection import chi2, mutual_info_classif

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Split data into features and target variable
X_ = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale numerical features to be non-negative (required for chi2)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_), columns=X_.columns)

# Use SelectKBest to select top features
selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X_scaled, y)


selected_features = X_.columns[selector.get_support()]
X = pd.DataFrame(X_new, columns=selected_features)



#train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=62,stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=62,class_weight="balanced",max_depth=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Initial Model Accuracy: {accuracy:.2f}')



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=600,
                               min_samples_split=2,
                               min_samples_leaf=4,
                               max_features='sqrt',
                               max_depth=30,
                               bootstrap=False,class_weight="balanced")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Initial Model Accuracy: {accuracy:.2f}')

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


smote = SMOTE(random_state=62)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

imputer = SimpleImputer(strategy='mean')

# Define pipeline
model1 = Pipeline([
    ('imputer', imputer),
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(n_estimators=600, min_samples_split=2,
                               min_samples_leaf=4, max_features='sqrt',
                               max_depth=30, bootstrap=False, random_state=42))
])

model1.fit(X_train_smote, y_train_smote)

predictions1 = model1.predict(X_test)

accuracy = accuracy_score(y_test, predictions1)
print(f'Accuracy after SMOTE: {accuracy:.2f}')
print(classification_report(y_test, predictions1))
print(confusion_matrix(y_test, predictions1))

from sklearn.model_selection import StratifiedKFold, cross_val_score


model = RandomForestClassifier(n_estimators=600, min_samples_split=2,
                               min_samples_leaf=4, max_features='sqrt',
                               max_depth=30, bootstrap=False, random_state=42)

stratified_kfold = StratifiedKFold(n_splits=5)

cv_scores = cross_val_score(model1, X_train_smote, y_train_smote, cv=stratified_kfold, scoring='accuracy')

print(f'Stratified Cross-Validation Accuracy for each fold: {cv_scores}')

print(f'Mean Stratified Cross-Validation Accuracy: {cv_scores.mean():.2f}')
print(f'Standard Deviation of Stratified Cross-Validation Accuracy: {cv_scores.std():.2f}')

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)
xgb_cv_scores = cross_val_score(xgb_model, X_train_smote,  y_train_smote, cv=5, scoring='accuracy')
print(f'XGBoost Cross-Validation Accuracy: {xgb_cv_scores.mean():.2f}')



import pickle
#pickle.dump(ensemble_model, open('ensemble_model.pkl', 'wb'))

pickle.dump(scaler, open('scaler.pkl', 'wb'))

#pickle label encoding
pickle.dump(label_encoder, open('label_encoding.pkl','wb'))

pickle.dump(xgb_model, open('model.pkl','wb'))