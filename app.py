import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Risk Prediction",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("Customer Risk Prediction Using SVM")
st.markdown("---")


# =========================================================
# INTRODUCTION
# =========================================================

st.header("Dashboard Introduction")

st.write(
    """
This web application predicts whether a customer is:

- Safe Customer
- Risky Customer

using Support Vector Machine (SVM).

Prediction is based on:

- Income
- Debt Ratio

Users can choose:

- Linear Kernel SVM
- Polynomial Kernel SVM
- RBF Kernel SVM
"""
)


# =========================================================
# CREATE DATASET
# =========================================================

np.random.seed(42)

N = 300

income = np.random.uniform(0, 10000, N)
debt_ratio = np.random.uniform(0, 100, N)

# 0 = Safe Customer
# 1 = Risky Customer

target = np.where(
    (income > 4000) & (debt_ratio < 50),
    0,
    1
)


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame({
    "Income": income,
    "Debt_Ratio": debt_ratio,
    "Target": target
})


# =========================================================
# PREPARE DATA
# =========================================================

X = df[["Income", "Debt_Ratio"]]
y = df["Target"]


# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# TRAIN MODELS
# =========================================================

linear_model = SVC(kernel='linear')
poly_model = SVC(kernel='poly', degree=3)
rbf_model = SVC(kernel='rbf')

linear_model.fit(X_train_scaled, y_train)
poly_model.fit(X_train_scaled, y_train)
rbf_model.fit(X_train_scaled, y_train)


# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("User Input")

income_input = st.sidebar.slider(
    "Income ($)",
    min_value=0,
    max_value=10000,
    value=5000
)

debt_input = st.sidebar.slider(
    "Debt Ratio (%)",
    min_value=0,
    max_value=100,
    value=50
)

kernel_choice = st.sidebar.selectbox(
    "Choose SVM Model",
    (
        "Linear Kernel SVM",
        "Polynomial Kernel SVM",
        "RBF Kernel SVM"
    )
)


# =========================================================
# SELECT MODEL
# =========================================================

if kernel_choice == "Linear Kernel SVM":
    model = linear_model

elif kernel_choice == "Polynomial Kernel SVM":
    model = poly_model

else:
    model = rbf_model


# =========================================================
# USER INPUT DATA
# =========================================================

user_data = np.array([[income_input, debt_input]])
user_data_scaled = scaler.transform(user_data)


# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(user_data_scaled)


# =========================================================
# SHOW RESULT
# =========================================================

st.header("Prediction Result")

if prediction[0] == 0:
    st.success("Result: Safe Customer")

else:
    st.error("Result: Risky Customer")


# =========================================================
# ACCURACY
# =========================================================

train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Accuracy",
        f"{train_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Testing Accuracy",
        f"{test_accuracy * 100:.2f}%"
    )


# =========================================================
# GRAPH VISUALIZATION
# =========================================================

st.header("Graph Visualization")

fig, ax = plt.subplots(figsize=(8, 6))

# Create Mesh Grid
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict Grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot Decision Boundary
ax.contourf(xx, yy, Z, alpha=0.3)

# Plot Safe Customer
ax.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    label="Safe Customer"
)

# Plot Risky Customer
ax.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    label="Risky Customer"
)

# Plot User Input
ax.scatter(
    user_data_scaled[0, 0],
    user_data_scaled[0, 1],
    marker='X',
    s=250,
    color='red',
    label='User Input'
)

ax.set_title(kernel_choice)
ax.set_xlabel("Income (Scaled)")
ax.set_ylabel("Debt Ratio (Scaled)")
ax.legend()

st.pyplot(fig)


# =========================================================
# DATASET PREVIEW
# =========================================================

st.header("Dataset Preview")

st.dataframe(df.head(10))


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.write("Developed using Streamlit and Scikit-Learn")