import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from fpdf import FPDF

# ============================================
# LOGIN SYSTEM
# ============================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.sidebar.title("🔐 Login System")

if not st.session_state.logged_in:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

    st.title("🎓 Student Placement Prediction System")
    st.info("Please login to continue.")
    st.stop()

# Logout Button
st.sidebar.success("Logged in successfully ✅")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ============================================
# LOAD DATA
# ============================================

data = pd.read_csv("data/placement.csv")

# Convert categorical Branch to numeric
data = pd.get_dummies(data, columns=["Branch"], drop_first=True)

X = data.drop("Placed", axis=1)
y = data["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ============================================
# NAVIGATION
# ============================================

st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Model Performance",
        "Feature Importance",
        "Confusion Matrix",
        "ROC Curve",
        "Prediction"
    ]
)

# ============================================
# DASHBOARD
# ============================================

if page == "Dashboard":
    st.title("📊 Placement Analytics Dashboard")

    st.subheader("Dataset Overview")
    st.dataframe(data)

    st.subheader("Placement Distribution")
    st.bar_chart(data["Placed"].value_counts())

    st.subheader("CGPA vs Placement")
    st.line_chart(data.groupby("CGPA")["Placed"].mean())

# ============================================
# MODEL PERFORMANCE
# ============================================

elif page == "Model Performance":
    st.title("🎯 Model Performance")
    st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# ============================================
# FEATURE IMPORTANCE
# ============================================

elif page == "Feature Importance":
    st.title("📌 Feature Importance")

    importance = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    st.pyplot(fig)

# ============================================
# CONFUSION MATRIX
# ============================================

elif page == "Confusion Matrix":
    st.title("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax_cm.text(j, i, cm[i][j], ha="center", va="center")

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

# ============================================
# ROC CURVE
# ============================================

elif page == "ROC Curve":
    st.title("📈 ROC Curve")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"AUC = {roc_auc:.2f}")

    st.pyplot(fig_roc)

# ============================================
# PREDICTION + PDF REPORT
# ============================================

elif page == "Prediction":
    st.title("🔮 Predict Placement")

    cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
    internships = st.number_input("Internships", 0, 5)
    aptitude = st.number_input("Aptitude Score", 0, 100)
    communication = st.number_input("Communication (1-10)", 1, 10)
    projects = st.number_input("Projects", 0, 10)
    certifications = st.number_input("Certifications", 0, 10)

    if st.button("Predict"):
        input_data = np.array([[cgpa, internships, aptitude,
                                communication, projects, certifications]
                               + [0]*(X.shape[1]-6)])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result_text = "Student is Likely to be Placed"
            st.success("🎉 " + result_text)
        else:
            result_text = "Student is Not Likely to be Placed"
            st.error("❌ " + result_text)

        # ===============================
        # PDF GENERATION
        # ===============================

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Student Placement Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"CGPA: {cgpa}", ln=True)
        pdf.cell(200, 10, txt=f"Internships: {internships}", ln=True)
        pdf.cell(200, 10, txt=f"Aptitude Score: {aptitude}", ln=True)
        pdf.cell(200, 10, txt=f"Communication: {communication}", ln=True)
        pdf.cell(200, 10, txt=f"Projects: {projects}", ln=True)
        pdf.cell(200, 10, txt=f"Certifications: {certifications}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Prediction Result: {result_text}", ln=True)
        pdf.cell(200, 10, txt=f"Model Accuracy: {accuracy*100:.2f}%", ln=True)

        pdf_output = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name="placement_report.pdf",
            mime="application/pdf"
        )