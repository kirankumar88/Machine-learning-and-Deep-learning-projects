  # PCA + K-Means Wine Segmentation

## Overview
This project demonstrates how **Principal Component Analysis (PCA)** improves **K-Means clustering** by reducing noise, eliminating multicollinearity, and enhancing cluster geometry in high-dimensional datasets.
Using a wine chemistry dataset with 13 physicochemical features, clustering performance is evaluated **before and after PCA** to quantify the real impact of dimensionality reduction.

---

## Dataset
- **Samples:** 178 wine instances  
- **Features:** 13 continuous chemical attributes  
- **Target:** Wine type (used only for validation and visualization)

---

## Workflow
1. Exploratory Data Analysis (EDA)
2. Feature standardization
3. Principal Component Analysis (PCA)
4. K-Means clustering on:
   - Original standardized feature space
   - PCA-reduced feature space
5. Quantitative evaluation and visual comparison

---

## Key Results

| Model | Silhouette Score | Davies–Bouldin Index |
|-----|-----------------|---------------------|
| K-Means (Original Features) | 0.28 | 1.38 |
| K-Means + PCA (3 Components) | **0.45** | **0.84** |

**Insight:**  
PCA delivers a clear and measurable uplift in clustering quality by removing redundancy and noise from correlated features.

---

## Why PCA Improves Clustering
- Eliminates multicollinearity (e.g., phenols and flavanoids)
- Compresses information into orthogonal components
- Improves alignment with K-Means’ spherical cluster assumption
- Enhances cluster separation and interpretability

---

## Techniques Used
- Exploratory Data Analysis (EDA)
- StandardScaler normalization
- Principal Component Analysis (PCA)
- K-Means clustering
- Silhouette score
- Davies–Bouldin index
- 2D and 3D PCA visualizations

---

## Practical Applications
- Customer and market segmentation
- Biological and chemical pattern discovery
- High-dimensional exploratory data analysis
- Preprocessing pipeline for unsupervised learning

---
