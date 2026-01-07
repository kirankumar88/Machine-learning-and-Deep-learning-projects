# Customer Segmentation Using K-Means and DBSCAN

## Overview
This project applies **unsupervised learning techniques** to segment airline customers based on their travel behavior and loyalty metrics. Using **K-Means** and **DBSCAN**, the analysis identifies meaningful customer groups that can support data-driven decision-making in marketing and customer retention strategies.

The workflow follows a structured data science approach: data preprocessing, exploratory data analysis (EDA), clustering, evaluation, and interpretation.

---

## Objective
The objective of this project is to explore and compare clustering algorithms—specifically **K-Means** and **DBSCAN**—to uncover hidden patterns in airline customer data. The goal is to evaluate how different clustering techniques group customers and to interpret the resulting segments in a business-relevant context.

---

## Dataset
The analysis uses the **EastWest Airlines customer dataset**, which contains customer-level information related to frequent flyer activity, mileage accumulation, and loyalty program engagement.

### Key Features
- Balance: Miles available for award travel  
- Qual_miles: Miles qualifying for elite status  
- Bonus_miles: Bonus miles earned from non-flight activities  
- Bonus_trans: Number of bonus transactions  
- Flight_miles_12mo: Flight miles in the last 12 months  
- Flight_trans_12: Flight transactions in the last 12 months  
- Days_since_enroll: Days since enrollment in the loyalty program  
- Award?: Indicator of award flight usage  

The dataset is well-suited for clustering due to its numerical structure and absence of predefined labels.

---

## Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Outlier detection
   - Feature scaling for distance-based clustering

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis using histograms and boxplots
   - Relationship analysis using scatter plots and correlation heatmaps

3. **Clustering Techniques**
   - **K-Means Clustering**
     - Optimal number of clusters determined using the Elbow Method
   - **DBSCAN**
     - Density-based clustering to identify core points and noise
     - Parameter tuning using epsilon and minimum samples

4. **Evaluation**
   - Internal validation using **Silhouette Score**
   - Qualitative comparison of clustering behavior

---

## Key Insights
- K-Means effectively segments customers into distinct behavioral groups but requires predefined cluster counts.
- DBSCAN identifies dense customer groups and isolates noise, making it useful for detecting outliers.
- Different algorithms reveal different structural perspectives of the same dataset.
- Customer segments can be interpreted in terms of loyalty, travel frequency, and engagement intensity.

---

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---


---

## Conclusion
This project demonstrates the practical application of unsupervised learning for customer segmentation. By comparing K-Means and DBSCAN, it highlights the importance of algorithm selection based on data characteristics and analytical objectives.

The results provide a foundation for extending the analysis toward targeted marketing, customer profiling, and retention strategies.

---

## Future Enhancements
- Add hierarchical clustering for comparison
- Perform dimensionality reduction (PCA) for visualization
- Translate clusters into actionable business personas

