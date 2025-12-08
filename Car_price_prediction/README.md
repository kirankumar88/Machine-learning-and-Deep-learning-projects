                                                                                    Car Price Prediction 

This repository presents a structured, analytics-driven approach to modeling used Toyota Corolla car prices using classical regression techniques. The project follows a disciplined machine-learning workflow—starting from data audit and preprocessing through to model development, regularization, and performance benchmarking.

The objective is straightforward:
Build a reliable, interpretable, and operationally robust pricing model suitable for analytical and decision-support environments.

🏗️ 1. Project Overview
Accurate pricing of used vehicles is a core requirement for automotive marketplaces, dealerships, and financial institutions. This project evaluates a curated set of vehicle attributes—including age, mileage, engine capacity, horsepower, vehicle weight, door configuration, and fuel type—to quantify their impact on market value.

The analysis delivers:
A clean and regression-ready dataset

A defensible feature-engineering pipeline

Multiple modeling strategies with diagnostic validation

A final recommended model optimized for accuracy and interpretability

🧹 2. Data Preparation & Quality Controls :
The dataset underwent a rigorous preprocessing workflow designed to ensure model stability and business relevance:

Key Steps
Column rationalization: Removed non-informative attributes (Cylinders, Gears).

Outlier management: 
Applied adaptive IQR-based capping to reduce the influence of extreme values while preserving natural variance.

Categorical engineering: One-hot encoding of Doors and Fuel_Type with defensible baseline selection.

Imbalance mitigation: Removed the Automatic column due to severe distribution skew.

Train/test allocation: Standard 80/20 split for unbiased performance estimation.

This produced a high-integrity dataset that meets linear modeling assumptions.

📊 3. Exploratory Data Analysis :
The EDA surfaced several economically consistent trends:

Age and KM exhibit a strong negative relationship with price—reflecting depreciation.

Weight and engine capacity (cc) correlate positively with price, consistent with higher-trim variants.

Fuel type demonstrates meaningful pricing differentials.

cc and Weight show material multicollinearity, influencing model architecture decisions.

These insights guided both feature selection and the subsequent regularization strategy.

🤖 4. Modeling Framework :
Three complementary regression models were developed and evaluated:

1️⃣ Multiple linear regression Ordinary Least Squares (OLS) :
Serves as the benchmark model.

Stable after removing multicollinear variables.

Achieves strong explanatory power (~85% variance explained).

2️⃣ Ridge Regression :
Controls coefficient inflation caused by correlated predictors.

Produces more stable, generalizable models.

3️⃣ Lasso Regression (Recommended Model) :
Performs embedded feature selection by shrinking weak predictors to zero.

Delivered the best overall performance across MAE, RMSE, and R².

Produces a compact and operationally efficient model.

🏆 5. Model Performance Overview :
Lasso Regression consistently outperformed OLS and Ridge, providing:

Highest R² (≈ 0.862)

Lowest prediction error (RMSE/MAE)

Most interpretable coefficient landscape

Given its accuracy, stability, and parsimony, Lasso is the recommended production-ready model for this dataset.

🧪 6. Diagnostic Validation :
All models underwent comprehensive statistical diagnostics:

Residual normality and symmetry: Confirmed

Homoscedasticity: No structural variance issues detected

Influential points: No observations exceeded standard risk thresholds

Multicollinearity: Effectively mitigated via feature reduction and regularization

The final model meets the quality standards expected in analytical modeling environments.

📁 7. Recommended Repository Structure
car-price-prediction/
│
├── data/
├── src/
├── figures/
├── notebooks/
├── outputs/
└── README.md


This structure supports maintainability, reproducibility, and auditability—core requirements for corporate analytics teams.

✅ 8. Key Business Insights :
Pricing is primarily driven by vehicle age, mileage, weight, and engine characteristics.

Regularization techniques materially improve model reliability in the presence of correlated engineering attributes.

A streamlined, Lasso-based regression model provides the optimal balance between interpretability and predictive power.

📌 9. Conclusion:
This project demonstrates a complete analytical modeling lifecycle aligned with industry best practices.
The resulting model is:

Statistically sound

Business-aligned

Generalization-ready

Interpretable for audit and stakeholder communication

The approach is fully extensible to broader vehicle pricing datasets and commercial applications.
