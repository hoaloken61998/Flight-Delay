# ✈️ Flight Delay Prediction using Machine Learning & Sampling Approaches

> This repository contains the full pipeline, experiments, and deployment scripts for predicting **flight delays** using machine learning.  
> The project is built on two reports:  
> - *ICE Conference Paper*: “Building a ML Model integrating In-depth Analysis and Sampling Approaches to predict Flight Delay”  
> - *Faculty Report*: “Xây dựng hệ thống dự đoán độ trễ chuyến bay cho người dùng sử dụng mô hình máy học tích hợp phương pháp lấy mẫu”  

Our goal is to create a **robust, interpretable, and deployable system** that supports airlines and passengers in managing delays.

---

## 1. Background & Overview

## 1. Background & Overview

Flight delays remain a major challenge for the aviation industry, leading to higher costs for airlines, lost time for passengers, and reduced overall efficiency. Traditional forecasting methods struggle to capture the complex factors behind delays, such as airline operations, airport congestion, scheduling, and weather conditions.  

This project applies machine learning to predict flight delays using large-scale data from the U.S. Bureau of Transportation Statistics (BTS). It combines both classification (delayed vs. on-time) and regression (delay duration) models while addressing class imbalance with sampling techniques like SMOTE. To ensure practical use, the solution is deployed through a Flask-based web interface, allowing users to receive real-time delay predictions from input flight details.

---

## 2. Data Structure

**Source:** U.S. Bureau of Transportation Statistics (BTS)  

**Key Features:**
- Flight schedule info → `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`  
- Airline & airports → `CARRIER`, `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID`  
- Delay components → `DEP_DELAY`, `TAXI_OUT`, `ARR_DELAY`  
- Distance metrics → `DISTANCE`  
- Temporal → `YEAR`, `QUARTER`, `DAY_OF_MONTH`, `DAY_OF_WEEK`

**Target Variables:**
- `STATUS` → Binary (Delayed = 1 if ARR_DELAY > 15 , else 0)  
- `ARR_DELAY` → Continuous (Minutes of arrival delay)  

**Dataset Scale:**
- ~300,000 flights  
- ~372 airports  
- Multiple airlines across U.S. domestic flights  

📊 Example:  
![Dataset Distribution](reports/figures/dataset_distribution.png)

---

## 3. Tech Stack

- **Programming Language:** Python 3.8+  
- **Data Analysis:** `pandas`, `numpy`  
- **Machine Learning Models:** `scikit-learn`, `xgboost`  
- **Resampling Methods:** `imbalanced-learn` (SMOTE variants)  
- **Model Explainability:** `shap`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Deployment:** `Flask` web framework  
- **Model Persistence:** `joblib`

---

## 4. Findings & Insights

### Exploratory Data Analysis (EDA)
- **Delay Distribution**:  
  - ~65% of flights were on-time, ~35% delayed → confirming a mild class imbalance.  
  - Most arrival delays ranged from **15–60 minutes**, with extreme outliers exceeding several hours.  
- **Temporal Trends**:  
  - Delays were more frequent during **evening hours** and at the **end of the week**.  
  - Seasonal spikes were observed in **summer (Q3)** and **holiday months**, consistent with traffic surges.  
- **Airline & Airport Effects**:  
  - Certain carriers had systematically higher delay rates.  
  - Congested airports (e.g., ATL, ORD) exhibited above-average delays.  
- **Correlation Analysis**:  
  - Strong correlation between `DEP_DELAY` and `ARR_DELAY` (departure delays propagate).  
  - Flight distance had only a weak relationship with delay duration.  

---

### Modeling & Performance Insights
1. **Class Imbalance Challenge**  
   - Raw datasets biased models toward predicting “on-time.”  
   - Applying **SMOTE variants** (SMOTEENN, SMOTETomek, SMOTE-RUS) improved recall for delayed flights and produced more balanced decision boundaries.  

2. **Model Comparisons**  
   - **Random Forest, Stacking + SMOTEENN, and Random Forest + SMOTETomek** consistently achieved **>80% precision for both classes**.  
   - Logistic Regression with **SMOTE-RUS** delivered robust performance (Accuracy ≈ 82%, ROC-AUC > 80%) while being computationally efficient.  
   - Regression models predicted delay minutes with **R² ≈ 0.87**, though high **MAPE (50–70%)** showed difficulty in estimating exact durations.  

3. **Explainability (SHAP Analysis)**  
   - Key predictive features:  
     - **Departure time** (CRS_DEP_TIME)  
     - **Origin & destination airports**  
     - **Airline carrier**  
     - **Departure delay (DEP_DELAY)**  
   - SHAP values highlighted how peak-time flights and congested hubs increased delay probability.  

---

### Key Takeaways
- **Trade-off in Model Selection**:  
  Complex ensemble models provided slightly higher accuracy, but **Logistic Regression** was ultimately chosen for deployment due to **speed, simplicity, and stability**.  

- **Two-Stage Prediction Pipeline**:  
  1. Classification → On-time vs. Delayed  
  2. If Delayed → Regression to estimate delay duration  

- **Practical Implications**:  
  - Regular retraining (e.g., monthly) is needed to adapt to **seasonal trends**.  
  - Airlines and airports can use SHAP-based insights to **target operational improvements** (e.g., staffing at peak hours, congestion management).  

---

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

