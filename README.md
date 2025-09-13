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

- **Temporal Patterns:** Evening departures (after 18:00) showed the highest probability of delay, and weekends (Friday–Sunday) had increased delays compared to weekdays. Seasonal peaks were observed in **July and December**.  

- **Airline & Airport Trends:** Major hubs such as **ATL** and **ORD** recorded above-average delays (above **40%** of flights delayed), while smaller airports had lower delay rates. Certain carriers consistently performed worse than others.  

- **Correlation Analysis:** The correlation between `DEP_DELAY` and `ARR_DELAY` exceeded **0.92**, highlighting that departure delays almost directly propagate to arrival delays. In contrast, flight distance had only weak correlation (< 0.1) with delay duration. 

---

### Faculty Report (2022–2023 Data)

- **Updated Delay Definition:** Delays were defined as flights arriving more than **15 minutes late**, reflecting operational reporting standards.  

- **Delay Distribution:** The imbalance persisted, with **28–30% of flights delayed** under the new definition.  

- **Temporal Patterns:** Peak delays occurred during summer months (June–August) and winter holidays. Evening flights still showed the highest delay probability, with Mondays and Fridays standing out among weekdays.  

- **Carrier & Airport Effects:** Some airlines showed improved punctuality compared to 2016–2017, though large hubs (ATL, ORD, DFW) continued to dominate delay counts.  

- **Propagation of Delays:** As with earlier years, `DEP_DELAY` remained the most influential predictor of `ARR_DELAY`, with correlation above **0.93**.  

---

### Key Insights

- Flight delays are **not random** but follow identifiable patterns related to **time of day, day of week, season, carrier, and airport congestion**.  
- Departure delay is the **single most important predictor**, with correlations consistently above **0.9** in both datasets.  
- Despite the six-year gap, both datasets share consistent patterns, suggesting that while airline performance and traffic volumes evolve, the **underlying causes of delays remain stable**.

## 5. Machine Learning Solution

Both reports applied a variety of machine learning techniques, combined with sampling strategies, to address class imbalance and evaluate predictive performance. The focus of the ICE report was on method comparison and interpretability, while the Faculty report emphasized building a practical deployment-ready system.

---

### ICE Conference Report (2016–2017 Data)

- **Classification Models Tested:** Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, and Stacking (ensembles).  
  Random Forest and Stacking consistently achieved high precision (>80% for both delayed and non-delayed classes), while Logistic Regression with SMOTE-RUS provided competitive results with faster computation.  
  *See Figure 4.1 – “Comparison of classification models”* (`docs/images/ICE/fig4_1_classification_comparison.png`).

- **Regression Models Tested:** Linear Regression, Ridge/Lasso, Random Forest, and XGBoost were used to predict arrival delay in minutes.  
  XGBoost achieved **R² ≈ 0.87** on the test set, but the Mean Absolute Percentage Error (MAPE) remained high (50–70%), showing that predicting the exact duration of delays is more difficult than classifying delay occurrence.  
  *See Figure 4.2 – “Regression performance comparison”* (`docs/images/ICE/fig4_2_regression_comparison.png`).

- **Sampling Techniques:** SMOTE, SMOTETomek, SMOTEENN, and SMOTE-RUS were applied. Balancing the dataset improved recall for the minority (delayed) class by more than 10 percentage points compared to training on the raw distribution.  
  *See Figure 4.3 – “Impact of sampling on model performance”* (`docs/images/ICE/fig4_3_sampling_effects.png`).

- **Interpretability:** SHAP analysis identified `DEP_DELAY`, `CRS_DEP_TIME`, and `CARRIER` as the top features influencing predictions. Flights departing during peak congestion hours and from busy hubs had markedly higher SHAP values.  
  *See Figure 4.4 – “SHAP summary plot for feature importance”* (`docs/images/ICE/fig4_4_shap_summary.png`).

---

### Faculty Report (2022–2023 Data)

- **Classification Models:** The same family of models was tested, with Logistic Regression + SMOTE-RUS selected for deployment due to its balance of accuracy (~82%), ROC-AUC (>80%), and computational efficiency in real-time inference.  
  *See Figure 4.6 – “Evaluation of classifiers with SMOTE-RUS”* (`docs/images/Faculty/fig4_6_classifier_eval.png`).

- **Regression Models:** Tree-based regressors again showed strong fit, with Random Forest reaching training R² > 0.95 but lower test R² (≈0.81–0.83), highlighting overfitting risks. Linear models generalized more consistently but with lower predictive power.  
  *See Figure 4.7 – “Comparison of regression models on test set”* (`docs/images/Faculty/fig4_7_regression_eval.png`).

- **Sampling:** The report confirmed that resampling improved detection of delayed flights. SMOTE-RUS was ultimately favored because it balanced the dataset efficiently without excessive computational cost.  
  *See Figure 4.8 – “Comparison of SMOTE variants”* (`docs/images/Faculty/fig4_8_sampling_variants.png`).

- **Interpretability:** SHAP analysis validated the earlier findings: `DEP_DELAY` dominated predictions, followed by scheduling features (`CRS_DEP_TIME`, `DAY_OF_WEEK`) and airline/airport identifiers. This confirmed that patterns observed in 2016–2017 remain valid in 2022–2023.  
  *See Figure 4.9 – “SHAP feature importance for deployed model”* (`docs/images/Faculty/fig4_9_shap.png`).

---

### Key Insights

- **Consistent Predictors:** Across both time periods, `DEP_DELAY` was the strongest predictor, with SHAP confirming its influence on arrival delay outcomes.  
- **Classification vs Regression:** Classifying delay occurrence is substantially more reliable than predicting exact delay duration.  
- **Model Selection:** While complex ensembles achieve the best offline metrics, Logistic Regression with SMOTE-RUS proved the most suitable for deployment, balancing accuracy, speed, and interpretability.  
- **Sampling Benefits:** Resampling techniques are essential; they increase recall for delayed flights and reduce bias toward on-time predictions.


📊 Example:  
![Model Comparison](reports/figures/model_comparison.png)

---

## 8. Repository Structure

