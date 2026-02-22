# # Employee Retention Prediction =============
# # This script predicts whether an employee will stay or leave the company


# # Step1 - Import Libraries =============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Step2 - Load Dataset ==================
df = pd.read_csv("aug_train.csv")
# print(df.head())

# # Step3 - Exploratory Data Analysis {'EDA-1'} ====================
# print(df.shape) # 19000+ Rows and 14 Columns 
# print(df.info())
# print(df.describe())

# # Step4 = Missing Value Check ============
# # print(df.isnull().sum())

# # Step5 - Drop Irrelevant Columns ==============
# df.drop(['enrollee_id', 'city'], axis=1, inplace=True)
# # print(df.isnull().sum())
# # print(df.columns)

# # Step6 - Handle Missing Values ===============
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].median())
# print(df.isnull().sum())

# # Step7 - EDA-2 =============== 
# # Target Variable Distribution Analysic --->
# sns.countplot(x='target', data=df)
# plt.title("Job Change Distribution")
# print(plt.show())

# # Feature vs Target Variable Analysis --->
# sns.countplot(x='gender', hue='target', data=df)
# plt.title("Gender vs Job Change")
# print(plt.show())

# # Experience vs Target Variable Analysis --->
# sns.countplot(x='experience', hue='target', data=df)
# plt.xticks(rotation=90)
# plt.title("Experience vs Job Change")
# print(plt.show())

# # Education Level vs Target Variable Analysis --->
# sns.countplot(x='education_level', hue='target', data=df)
# plt.xticks(rotation=45)
# plt.title("Education Level vs Job Change")
# print(plt.show())


# # Step8 - Encode Categorical Columns ==============
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, drop_first=True)
# print(f'After Label Encoding, new shape: {df.shape}')

# # Step9 - Feature & Target Split ==============
# Feature & Target Split
X = df.drop('target', axis=1)
y = df['target']

# --- Step 10: Boosting Random Forest Performance --->
# Imbalanced Data Handling using SMOTE
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# # Step11 - Train Test Split ==============
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)


# # Step12 - Feature Scaling ONLY for Logistic Regression =============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print('Feature scaling completed.')

# --- Step13: Model Building & Evaluation ---> 

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
# print("Logistic Regression Accuracy:", lr_acc)
# print(classification_report(y_test, lr.predict(X_test_scaled)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, lr.predict(X_test_scaled)))

# 2. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
# print("Decision Tree Accuracy:", dt_acc)
# print(classification_report(y_test, dt.predict(X_test)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, dt.predict(X_test)))

# 3. Random Forest (Optimized for Peak Performance)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=18, 
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
# print("Random Forest Accuracy:", rf_acc)
# print(classification_report(y_test, rf.predict(X_test)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, rf.predict(X_test)))

# --- Step14: Comparison & Performance ---
# print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
# print(f"Decision Tree Accuracy:       {dt_acc:.4f}")
# print(f"Random Forest Accuracy:       {rf_acc:.4f}")

# Visualization to prove Random Forest is Best
# models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
# accuracies = [lr_acc, dt_acc, rf_acc]
# plt.figure(figsize=(10, 5))
# sns.barplot(x=models, y=accuracies, palette='viridis')
# plt.ylim(0, 1.0)
# plt.title('Model Comparison: Random Forest vs Others')
# plt.ylabel('Accuracy Score')
# print(plt.show())

# print("\n--- Final Best Model Report (Random Forest) ---")
# print(classification_report(y_test, rf.predict(X_test)))

# # Step15 - ROC-AUC Curve Visualization ==============
from sklearn.metrics import roc_auc_score, roc_curve
# rf_probs = rf.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, rf_probs)
# fpr, tpr, _= roc_curve(y_test, rf_probs)
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], 'k--') 
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-AUC Curve for Random Forest')
# plt.legend()
# print(plt.show())

# # Step16 - Feature Importance Visualization ==============
# feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
# feature_importances.sort_values(ascending=False).head(10).plot(kind='bar')
# plt.title("Top 10 Feature Importances from Random Forest")
# print(plt.show())

# Step17 - Save the Best Model ==============
import pickle
pickle.dump(rf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
# print("Model and Scaler saved successfully.")

# --- End of Employee Retention Prediction Script --->

