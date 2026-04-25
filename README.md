# 🛡️ ChurnGuard AI: Customer Retention Predictor

ChurnGuard AI is a machine learning application designed to predict the likelihood of customer churn with high precision. Built with a **Gradient Boosting** architecture, the model achieves a **0.91 ROC-AUC** score, making it a reliable tool for business decision-making.

## 🚀 Key Features
- **High Performance:** Optimized with Log transformations and custom Feature Engineering.
- **Dynamic Thresholding:** Balanced at **0.3695** to maximize F1-Score (0.70) and Recall (0.77).
- **Interactive UI:** A minimalist and fast web interface built with **Streamlit**.
- **Real-time Prediction:** Instant churn probability analysis based on customer demographics and billing.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tugcenkrdnz/Predict-Customer-Churn.git
   
   cd churnguard-ai

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run app.py


**📊 Model Performance**
ROC-AUC: 0.91

F1-Score: 0.70

Optimal Threshold: 0.3695

Top Features: Electronic Check payment, Contract Score, and Tenure.

**🧰 Technologies Used**
Python (Pandas, Numpy, Scikit-learn)

Gradient Boosting (XGBoost/LightGBM logic)

Streamlit (Web Interface)

Joblib (Model Serialization)

**Developed as part of a professional AI portfolio focusing on Customer Analytics.**