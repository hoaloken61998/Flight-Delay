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

Before building predictive models, both reports carried out detailed exploratory data analysis (EDA) to understand the nature of flight delays.  
This section summarizes key findings and points to the original figures included in the reports. For easier access, the figures are also stored in the repository under `docs/images/`.

---

### ICE Conference Report (2016–2017 Data)

- **Delay Distribution:** Around **34% of flights were delayed**, confirming significant class imbalance.  
  *See Figure 3.1 in ICE Report – “Distribution of delayed vs. non-delayed flights”* (`docs/images/ICE/fig3_1_delay_distribution.png`).

- **Temporal Patterns:** Evening departures (after 18:00) showed the highest probability of delay, and weekends (Friday–Sunday) had increased delays compared to weekdays. Seasonal peaks were observed in **July and December**.  
  *See Figure 3.2 – “Monthly delay trends”* (`docs/images/ICE/fig3_2_monthly_trends.png`).

- **Airline & Airport Trends:** Major hubs such as **ATL** and **ORD** recorded above-average delays (above **40%** of flights delayed), while smaller airports had lower delay rates. Certain carriers consistently performed worse than others.  
  *See Figure 3.3 – “Carrier delay rate comparison”* (`docs/images/ICE/fig3_3_carrier_delays.png`).

- **Correlation Analysis:** The correlation between `DEP_DELAY` and `ARR_DELAY` exceeded **0.92**, highlighting that departure delays almost directly propagate to arrival delays. In contrast, flight distance had only weak correlation (< 0.1) with delay duration.  
  *See Figure 3.4 – “Correlation heatmap”* (`docs/images/ICE/fig3_4_corr_heatmap.png`).

---

### Faculty Report (2022–2023 Data)

- **Updated Delay Definition:** Delays were defined as flights arriving more than **15 minutes late**, reflecting operational reporting standards.  

- **Delay Distribution:** The imbalance persisted, with **28–30% of flights delayed** under the new definition.  
  *See Figure 3.2 in Faculty Report – “Distribution of ARR_DELAY before removing outliers”* (`docs/images/Faculty/fig3_2_arr_delay.png`).

- **Temporal Patterns:** Peak delays occurred during summer months (June–August) and winter holidays. Evening flights still showed the highest delay probability, with Mondays and Fridays standing out among weekdays.  
  *See Figure 3.1 – “Distribution of CRS_ELAPSED_TIME and DISTANCE”* (`docs/images/Faculty/fig3_1_time_distance_distribution.png`).

- **Carrier & Airport Effects:** Some airlines showed improved punctuality compared to 2016–2017, though large hubs (ATL, ORD, DFW) continued to dominate delay counts.  
  *See Figure 4.2 – “Classification model evaluations by carrier and airport”* (`docs/images/Faculty/fig4_2_carrier_airport_eval.png`).

- **Propagation of Delays:** As with earlier years, `DEP_DELAY` remained the most influential predictor of `ARR_DELAY`, with correlation above **0.93**.  
  *See Figure 4.9 – “Evaluation based on 4 methods for test set”* (`docs/images/Faculty/fig4_9_eval_test.png`).

---

### Key Insights

- Flight delays are **not random** but follow identifiable patterns related to **time of day, day of week, season, carrier, and airport congestion**.  
- Departure delay is the **single most important predictor**, with correlations consistently above **0.9** in both datasets.  
- Despite the six-year gap, both datasets share consistent patterns, suggesting that while airline performance and traffic volumes evolve, the **underlying causes of delays remain stable**.


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

