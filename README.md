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

## 4. Findings & Insights (Exploratory Data Analysis)

Before building predictive models, both reports carried out detailed exploratory data analysis (EDA) to understand the flight delay problem.

### ICE Conference Report (2016–2017 Data)
- **Delay Distribution:** Roughly two-thirds of flights were on-time, while about one-third were delayed, confirming class imbalance in the dataset.  
- **Temporal Patterns:** Delays were more frequent during evening hours and toward the end of the week. Seasonal peaks appeared in summer and holiday months.  
- **Airline & Airport Trends:** Certain carriers consistently showed higher delay rates, and major hubs such as ATL and ORD recorded above-average delays due to congestion.  
- **Correlation Analysis:** Strong correlation observed between `DEP_DELAY` and `ARR_DELAY`, highlighting how departure delays propagate to arrivals. Flight distance showed little influence on delay duration.  

### Faculty Report (2022–2023 Data)
- **Updated Delay Definition:** Delays were defined as flights arriving more than **15 minutes late**, reflecting operational reporting standards.  
- **Delay Distribution:** Similar imbalance remained, with the majority of flights on-time but a significant minority delayed.  
- **Recent Temporal Trends:** Delays were concentrated in peak travel periods (summer and holidays), with noticeable increases in evening flights and Mondays/Fridays.  
- **Carrier & Airport Effects:** Some airlines demonstrated improved punctuality compared to the 2016–2017 dataset, while large airports still accounted for most delays.  
- **Propagation of Delays:** As with earlier years, `DEP_DELAY` was the most influential factor in predicting `ARR_DELAY`.  

---

### Key Insights
- Flight delays are **not random**; they follow clear patterns related to **time of day, day of week, season, carrier, and airport congestion**.  
- Departure delays strongly predict arrival delays, suggesting operational improvements at the departure stage could significantly reduce overall delay rates.  
- Despite the six-year gap, the two datasets share consistent patterns, indicating that while the scale of delays changes, the underlying causes remain stable over time.


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

