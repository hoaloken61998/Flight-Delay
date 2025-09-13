# ✈️ Flight Delay Prediction – Machine Learning & Sampling Approaches

> A project integrating **in-depth analysis**, **resampling methods**, and **machine learning** to predict flight delays.  
> Built from our ICE Conference paper and Faculty report.

---

## 1. Background & Overview

Flight delays cause huge operational and economic costs in the aviation industry.  
This project develops both **classification (delayed vs. on-time)** and **regression (minutes delayed)** models.  

We integrate **sampling techniques** to address class imbalance, and deploy the solution via a **Flask web interface**.

📑 References:  
- [ICE Conference Paper](./reports/ICE_paper.pdf)  
- [Faculty Report (Vietnamese)](./reports/Faculty_Report.docx)

---

## 2. Data Structure

- **Source:** Bureau of Transportation Statistics (BTS)  
- **Features:** `Quarter`, `DayOfWeek`, `Carrier`, `Origin`, `Destination`, `CRS_DEP_TIME`, `DISTANCE`, etc.  
- **Target Variables:**  
  - `STATUS` → Delayed / Not Delayed  
  - `ARR_DELAY` → Minutes of delay  

📊 Dataset scale: ~300k flight records across ~372 airports.

---

## 3. Tech Stack

- **Language:** Python 3.8+  
- **Libraries:** `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `flask`, `joblib`  
- **Deployment:** Flask API + simple web interface  

---

## 4. Key Findings

- **Class Imbalance** → solved with SMOTE variants (SMOTETomek, SMOTE-RUS).  
- **Best classifier** (tradeoff speed/accuracy): Logistic Regression + SMOTE-RUS.  
- **Performance:**  
  - Accuracy: > 82%  
  - ROC-AUC: > 80%  
- **Regression:** Tree-based models reached R² ≈ 0.87 (but risk overfitting).  

📌 Example:  
![Confusion Matrix](reports/figures/confusion_matrix.png)  
![ROC Curve](reports/figures/roc_curve.png)

---

## 5. Insights

- 🚦 **Operational Insight:** A two-step system is best — classify delayed vs. on-time first, then predict delay minutes only for delayed flights.  
- 🎯 **Model Choice:** High-performing ensemble models exist, but Logistic Regression was chosen for **real-time inference**.  
- 📊 **Explainability:** SHAP analysis highlights **departure time**, **origin airport**, and **carrier** as strong predictors.  

📌 Example:  
![SHAP Feature Importance](reports/figures/shap_feature_importance.png)

---

## 6. Recommendations

- Regularly **retrain models** (monthly or after major schedule changes).  
- Use **SMOTE-RUS** during training, but **evaluate on original distribution** to reflect reality.  
- Add **drift monitoring** and **logging** in the Flask app for continuous improvement.  

---

## 7. Machine Learning Solution

- **Classification models tested:** Logistic Regression, Random Forest, XGBoost, SVM, Stacking.  
- **Regression models tested:** Linear Regression, Ridge/Lasso, Random Forest, XGBoost.  
- **Evaluation metrics:**  
  - Classification → Accuracy, Precision, Recall, F1, ROC-AUC.  
  - Regression → MAE, RMSE, R², MAPE.  
- **Interpretability:** SHAP used for feature importance & local explanations.  

---

## 8. Repository Structure

