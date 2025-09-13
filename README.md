# ✈️ Flight Delay Prediction using Machine Learning & Sampling Approaches

> This repository contains the full pipeline, experiments, and deployment scripts for predicting **flight delays** using machine learning.  
> The project is built on two reports:  
> - *ICE Conference Paper*: “Building a ML Model integrating In-depth Analysis and Sampling Approaches to predict Flight Delay”  
> - *Faculty Report*: “Xây dựng hệ thống dự đoán độ trễ chuyến bay cho người dùng sử dụng mô hình máy học tích hợp phương pháp lấy mẫu”  

Our goal is to create a **robust, interpretable, and deployable system** that supports airlines and passengers in managing delays.

---
## 1. Background & Overview

Flight delays remain a major challenge for the aviation industry, leading to higher costs for airlines, lost time for passengers, and reduced overall efficiency. Traditional forecasting methods often fail to capture the many interacting factors behind delays, such as airline operations, airport congestion, scheduling, and weather conditions.  

This repository brings together two complementary research efforts that use **different datasets and periods**:  

- **ICE Conference Paper (2016–2017 data):** Focused on building robust machine learning models with integrated sampling approaches. The emphasis was on comparing algorithms and resampling strategies to handle class imbalance and to evaluate model interpretability with SHAP.  
- **Faculty Report (2022–2023 data):** Extended the approach to a more recent dataset and prioritized system deployment. This report emphasized building a practical web interface with Flask and selecting a model that balances accuracy and real-time performance.  

Together, these two works show how techniques evolve over time: from methodological exploration (2016–2017) to real-world system integration (2022–2023).

---

## 2. Data Structure

**Sources:** U.S. Bureau of Transportation Statistics (BTS)  

- **ICE Conference Report:** Flight data from **2016 and 2017** covering hundreds of thousands of domestic flights across ~372 airports. Target variable was `STATUS` (delayed if ARR_DELAY > 0).  
- **Faculty Report:** Flight data from **2022 and 2023**, reflecting more current patterns in airline performance and passenger demand. Target variable was also `STATUS`, but with a threshold of **ARR_DELAY > 15 minutes** to align with operational standards for reporting delays.  

**Key Features Across Both Reports:**
- Flight schedule → `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`  
- Airline & airports → `CARRIER`, `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID`  
- Delay components → `DEP_DELAY`, `TAXI_OUT`, `ARR_DELAY`  
- Distance metrics → `DISTANCE`  
- Temporal → `YEAR`, `QUARTER`, `DAY_OF_MONTH`, `DAY_OF_WEEK`

---

## 4. Findings & Insights

- **2016–2017 (ICE Report):**  
  - Focused on testing a wide range of models and resampling strategies.  
  - Found that ensemble models (Random Forest, Stacking, XGBoost) performed best in accuracy, but Logistic Regression with SMOTE-RUS remained competitive and interpretable.  
  - Demonstrated that SHAP values can highlight important predictors such as departure time and carrier.  

- **2022–2023 (Faculty Report):**  
  - Applied the methodology to a newer dataset with stricter definitions of delay (>15 minutes).  
  - Prioritized a model suitable for deployment: Logistic Regression with SMOTE-RUS, offering ~82% accuracy and ROC-AUC above 80%.  
  - Built and tested a Flask-based system, confirming the model’s usability in real-time prediction scenarios.  


## 5. Recommendations

- 📦 **Deployment Strategy:**  
  - Use Logistic Regression (SMOTE-RUS) for real-time predictions.  
  - Retrain monthly to adapt to seasonal trends and flight pattern shifts.  

- 📉 **Evaluation Protocol:**  
  - Always train on balanced datasets (SMOTE variants)  
  - Always test on **original distribution** to reflect reality  

- 🛠 **Future Improvements:**  
  - Integrate weather & airport traffic data  
  - Explore deep learning for time-series patterns  
  - Expand beyond U.S. flights for generalization  

---

## 6. Machine Learning Solution

**Models tested:**
- Classification → Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Stacking  
- Regression → Linear Regression, Ridge/Lasso, Random Forest, XGBoost  

**Sampling techniques:**
- SMOTE  
- SMOTETomek  
- SMOTEENN  
- SMOTE-RUS  

**Evaluation Metrics:**
- Classification → Accuracy, Precision, Recall, F1, ROC-AUC  
- Regression → MAE, RMSE, R², MAPE  

**Interpretability:**
- SHAP values to identify and rank influential features  

📊 Example:  
![Model Comparison](reports/figures/model_comparison.png)

---

## 8. Repository Structure

