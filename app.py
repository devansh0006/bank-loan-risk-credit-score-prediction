import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Streamlit App Title
st.title("Bank Loan Risk Analysis & Credit Score Prediction")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Loan_Data.csv')
    return df

df = load_data()

# Data Overview Section
st.write("## Data Overview")
st.write(f"Dataset contains {df.shape[0]} records with {df.shape[1]} features")
st.write("### First 5 rows:")
st.write(df.head())

# Data Analysis Section
st.write("## Exploratory Data Analysis")

# Default rate
default_rate = df['default'].mean()
st.write(f"### Default Rate: {default_rate:.2%}")

# Numeric features distribution
numeric_features = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                   'total_debt_outstanding', 'income', 'years_employed', 'fico_score']

st.write("### Feature Distributions")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, feature in enumerate(numeric_features):
    sns.histplot(df[feature], ax=axes[i//3, i%3], kde=True)
    axes[i//3, i%3].set_title(feature)
plt.tight_layout()
st.pyplot(fig)

# Correlation matrix
st.write("### Feature Correlation Matrix")
corr_matrix = df[numeric_features + ['default']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature selection
features = numeric_features
target = 'default'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training Section
st.write("## Model Training")

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
}

metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score,
    "AUC": roc_auc_score
}

results = {}

for name, model in models.items():
    if name == "Random Forest":
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probas = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        "model": model,
        "predictions": preds,
        "probabilities": probas
    }

# Display model performance
st.write("### Model Performance Comparison")

performance_data = []
for model_name, result in results.items():
    row = {"Model": model_name}
    for metric_name, metric_func in metrics.items():
        if metric_name == "AUC":
            score = metric_func(y_test, result["probabilities"])
        else:
            score = metric_func(y_test, result["predictions"])
        row[metric_name] = f"{score:.4f}"
    performance_data.append(row)

performance_df = pd.DataFrame(performance_data)
st.table(performance_df)

# Select best model (XGBoost)
best_model = results["XGBoost"]["model"]
best_scaler = scaler

# Feature Importance
st.write("### Feature Importance (XGBoost)")
importance = best_model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
ax.set_title("Feature Importance for Default Prediction")
st.pyplot(fig)

# Confusion Matrix
st.write("### Confusion Matrix (XGBoost)")
cm = confusion_matrix(y_test, results["XGBoost"]["predictions"])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Prediction Section
st.write("## Customer Risk Assessment")

# Create input form
st.write("### Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    credit_lines = st.number_input("Number of Credit Lines Outstanding", 
                                  min_value=0, max_value=20, value=1)
    loan_amt = st.number_input("Loan Amount Outstanding ($)", 
                              min_value=0, value=5000)
    total_debt = st.number_input("Total Debt Outstanding ($)", 
                                min_value=0, value=10000)

with col2:
    income = st.number_input("Annual Income ($)", 
                            min_value=0, value=50000)
    years_employed = st.number_input("Years Employed", 
                                    min_value=0, max_value=50, value=5)
    fico_score = st.number_input("FICO Score", 
                                min_value=300, max_value=850, value=650)

# Prediction button
if st.button("Assess Risk and Calculate Credit Score"):
    # Prepare input data
    input_data = pd.DataFrame([[
        credit_lines, loan_amt, total_debt, income, years_employed, fico_score
    ]], columns=features)
    
    # Scale input
    input_scaled = best_scaler.transform(input_data)
    
    # Make prediction
    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1] * 100
    
    # Calculate credit score (simplified model)
    base_score = fico_score
    income_adjustment = min(100, income / 5000)  # $50k income = +100 points
    debt_adjustment = -min(150, total_debt / 1000)  # $100k debt = -150 points
    employment_adjustment = min(50, years_employed * 2)  # 25 years = +50 points
    
    credit_score = int(base_score + income_adjustment + debt_adjustment + employment_adjustment)
    credit_score = max(300, min(850, credit_score))  # Ensure within valid range
    
    # Display results
    st.write("## Risk Assessment Results")
    
    if prediction == 1:
        st.error(f"High Risk: {probability:.1f}% probability of default")
    else:
        st.success(f"Low Risk: {probability:.1f}% probability of default")
    
    st.write(f"### Estimated Credit Score: {credit_score}")
    
    # Risk interpretation
    if credit_score >= 720:
        st.success("Excellent Credit - Likely to qualify for best rates")
    elif credit_score >= 690:
        st.info("Good Credit - May qualify for competitive rates")
    elif credit_score >= 630:
        st.warning("Fair Credit - May qualify but with higher rates")
    else:
        st.error("Poor Credit - May have difficulty qualifying")
    
    # Visual indicators
    st.write("### Risk Indicators")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Default Risk'], [probability], color='red' if probability > 30 else 'orange' if probability > 15 else 'green')
    ax.set_xlim(0, 100)
    ax.set_title('Default Probability')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Credit Score'], [credit_score], color='green' if credit_score >= 720 else 'lightgreen' if credit_score >= 690 else 'yellow' if credit_score >= 630 else 'red')
    ax.set_xlim(300, 850)
    ax.set_title('Credit Score')
    st.pyplot(fig)

# Add some explanations
st.write("""
### About This Analysis
- **Default Prediction**: The model predicts whether a customer is likely to default on their loan based on their financial profile.
- **Credit Score**: The estimated credit score is calculated using a simplified model that considers FICO score, income, debt, and employment history.
- **Interpretation**: 
  - Default risk > 30% is considered high risk
  - Credit scores range from 300-850 (higher is better)
""")