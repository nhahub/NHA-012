# **Exploratory Data Analysis (EDA) Phase Report**

## **💡 Phase Context**

This document provides detailed documentation for the Exploratory Data Analysis (**EDA**) phase of a larger data science project focused on forecasting weekly retail sales.

This phase is critical for moving from raw, separate datasets to a cohesive, cleaned, and insightful data structure. Its primary objectives are to:

* **Validate Data Quality** (e.g., handling missing values, outliers).  
* **Unify Data Sources** into single, time-series-consistent DataFrames.  
* **Generate Initial Insights** into sales drivers (e.g., holidays, external factors) to inform feature engineering and model selection in subsequent project phases.

## **🛠️ Data Preparation and Preprocessing**

### **1\. Data Merging and Initial Cleaning**

The initial step involved integrating the separate datasets to create comprehensive records for every **Store**, **Department**, and **Date**.

* **Merging Strategy:** The features and stores tables were joined first on the Store ID. This composite table was then left-merged with the train and test tables using a combined key: **Store**, **Date**, and **IsHoliday**. This specific key ensures external factors (like fuel price and CPI) are correctly aligned with the corresponding sales and holiday status.  
* **Time-Series Consistency:** All date columns were explicitly converted to datetime objects. Both the train\_merged and test\_merged DataFrames were then sorted by Store, Dept, and Date. This chronological ordering is mandatory for later time-series analysis and the creation of lag features.  
* **Data Validation:** Rows with **Weekly\_Sales less than or equal to zero** were removed from the training data, as negative sales figures are typically considered invalid or indicative of non-standard transactions that could skew predictive modeling. Duplicate records were also removed.

### **2\. Missing Value Imputation and Indication**

The strategy for handling missing data, particularly in the MarkDown columns, was two-pronged to maximize information retention.

* **Rationale for MarkDown Imputation:** The markdown fields often contain missing values, likely indicating that **no markdown promotion was running** that week, meaning a value of zero.  
* **Indicator Features:** For each MarkDown column, a **binary indicator column** (e.g., MarkDown1\_is\_missing) was created (value of 1 if missing, 0 if present). This allows a model to learn from both the value of the markdown and the fact that a promotion was absent.  
* **Final Imputation:** After creating the indicators, all remaining **NaN** values in the entire dataset (including the markdown columns themselves) were replaced with **0**. This ensures a completely clean, numerical dataset for algorithm consumption.

### **3\. Feature Transformation and Encoding**

To satisfy the requirements of certain predictive models (especially linear ones), we transformed the sales target and encoded categorical features.

| Feature | Transformation Applied | Detailed Rationale |
| :---- | :---- | :---- |
| **Weekly\_Sales** | Cube Root Transformation (**Weekly\_Sales\_cbrt**) | The raw Weekly\_Sales distribution was highly **positively skewed**. Applying a cube root (np.cbrt) transformation helped push the distribution toward a more normal, or Gaussian-like, shape. This often improves the performance and reliability of models that assume normally distributed errors. |
| **Type (A, B, C)** | One-Hot Encoding | The Type column, being **nominal categorical data**, was converted into three mutually exclusive binary columns. This prevents the model from incorrectly inferring an ordinal relationship. |
| **Outlier Detection** | IQR Method on Weekly\_Sales\_cbrt | Using the **IQR method** on the transformed sales data is more robust. It identified 3,005 outliers (**0.72%** of data), informing us that extreme values exist and may warrant a robust model or specific treatment in later phases. |

## **📊 Key Analytical Findings**

### **1\. Temporal Analysis**

* **Strong Seasonality:** The aggregated weekly sales plot revealed pronounced **seasonality** repeating across the years, confirming the need for time-based features in modeling.  
* **Quantified Holiday Impact:** The separate time series for holiday sales clearly showed distinct, massive spikes compared to non-holiday sales. The **IsHoliday** flag is confirmed as one of the most significant categorical predictors.  
* **CPI Trend:** The **Consumer Price Index (CPI)** showed an increasing, near-linear upward trend over the observation period, indicating general inflationary pressure, which serves as a crucial economic context.

### **2\. Feature Relationship Analysis**

| Feature | Relationship to Weekly\_Sales | Implications for Modeling |
| :---- | :---- | :---- |
| **Size** | Moderate Positive Correlation (**≈ 0.25**) | A valuable feature: the model should strongly leverage **store size** to explain sales variance. |
| **Temperature** | Weak/Non-linear | The correlation is weak, and the binned plot suggests sales dip slightly at both high and low temperature extremes. This indicates a **non-linear relationship** that simple linear models may miss. |
| **Fuel Price** | Subtle Inverse Correlation | A slight negative correlation suggests rising fuel costs may marginally dampen consumer spending. |
| **CPI / Unemployment** | Negligible Linear Correlation | These features, in their current raw form, are not linearly predictive of sales and may not be useful without significant **feature engineering** (e.g., deriving rate of change or comparing them against historical norms). |
| **MarkDown Inter-Correlation** | High correlation among MarkDown features. | This warns of potential **multicollinearity** issues. If we use all MarkDown variables in a highly interpretable model (like linear regression), we may need to select a subset or use dimensionality reduction. Tree-based models are generally unaffected. |

## **📝 Deliverables and Transition**

This EDA phase has successfully delivered:

* **Cleaned and Preprocessed DataFrames:** train\_merged and test\_merged, ready for feature engineering.  
* **Target Transformation:** The **Weekly\_Sales\_cbrt** feature is prepared for model training.  
* **Key Insights:** Confirmation of high-impact factors (**Holidays**, **Size**) and identification of complex or weak relationships that require further feature engineering.