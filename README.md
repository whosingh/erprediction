# 👩‍💼 Employee Retention Prediction

This project predicts whether an employee will stay with the company or leave using Machine Learning models.  
It includes data preprocessing, exploratory data analysis (EDA), handling imbalanced data, model comparison, and saving the best-performing model.

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations.  
This project builds a predictive system that:

- Analyzes employee data
- Identifies important features influencing job change
- Compares multiple ML models
- Selects the best-performing model
- Saves the trained model for future use

---

## 📂 Dataset Information

- **Dataset File:** `aug_train.csv`
- **Rows:** ~19,000+
- **Columns:** 14
- **Target Column:** `target`
  - `0` → Employee stays
  - `1` → Employee leaves

---

## 🛠️ Technologies & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Pickle

---

## 🔄 Project Workflow

### 1️⃣ Data Loading
- Load dataset using Pandas.

### 2️⃣ Data Cleaning
- Dropped irrelevant columns: `enrollee_id`, `city`
- Missing values handled:
  - Categorical → Mode
  - Numerical → Median

### 3️⃣ Exploratory Data Analysis (EDA)
- Target distribution
- Gender vs Target
- Experience vs Target
- Education level vs Target

### 4️⃣ Encoding Categorical Variables
- Label Encoding
- One-Hot Encoding (`get_dummies()`)

### 5️⃣ Handling Imbalanced Data
- Used **SMOTE** to balance target classes.

### 6️⃣ Train-Test Split
- 80% Training
- 20% Testing
- Stratified sampling

### 7️⃣ Feature Scaling
- Applied `StandardScaler` (for Logistic Regression only)

### 8️⃣ Models Trained

- Logistic Regression
- Decision Tree
- Random Forest (Optimized)

---

## 🏆 Best Model: Random Forest

Optimized Parameters:

- `n_estimators = 300`
- `max_depth = 18`
- `min_samples_split = 5`
- `max_features = 'sqrt'`
- `n_jobs = -1`
- `random_state = 42`

Random Forest achieved the highest accuracy among all models.

---

## 📊 Model Evaluation

- Accuracy Score
- Classification Report
- Confusion Matrix
- ROC-AUC Curve
- Feature Importance Visualization

---

## 💾 Saved Files

After training, the following files are generated:

```
model.pkl
scaler.pkl
```

These files can be used later for deployment or prediction.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/employee-retention-prediction.git
cd employee-retention-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Script

```bash
python employee_retention.py
```

---

## 📈 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Deployment with Flask or FastAPI
- Streamlit dashboard
- Try XGBoost or LightGBM

---

## 🎯 Business Impact

This model helps HR teams:

- Identify employees at risk of leaving
- Improve retention strategies
- Reduce hiring & training costs
- Make data-driven HR decisions

---

## 👨‍💻 Author

Shanu Singh  
Machine Learning Enthusiast
