🧠 Drug Response Classification using Support Vector Machine (SVM)

A complete machine learning workflow to classify drug response (0 = No Response, 1 = Positive Response) using clinical and biochemical features.
This project demonstrates end-to-end ML, including EDA, preprocessing, SVM modeling, hyperparameter tuning, evaluation, and interpretation.

📌 Project Overview

Pharmaceutical companies invest heavily in clinical trials, but not all patients respond to a drug the same way.
This project uses SVM classification to predict drug response based on patient biomarker data.

The workflow includes:

Exploratory Data Analysis (EDA)

Visualization of feature distributions & relationships

Data preprocessing

SVM model training

Kernel comparison (Linear, Polynomial, RBF)

Hyperparameter tuning

Model evaluation (Accuracy, Precision, Recall, F1-score, AUC)

Final insights & conclusion

📊 Dataset Description

The dataset contains 500 patient entries with 5 clinical features:

Feature	Meaning
Drug Dosage (mg)	Administered drug amount
Systolic BP (mmHg)	Blood pressure indicator
Heart Rate (BPM)	Cardiovascular response
Liver Toxicity Index (U/L)	Safety biomarker
Blood Glucose (mg/dL)	Metabolic response
Drug Response (0/1)	Target variable (0 = No Response, 1 = Positive Response)

Dataset file:
📁 dataset/Drug_Response_Classification_data.csv

🔍 Exploratory Data Analysis (EDA)

The notebook includes:

Histograms to understand feature distributions

Boxplots to inspect outliers

Correlation Heatmap for feature relationships

Pairplots showing non-linear separability

Scatter plots for biomarker interactions

Class distribution plot to check dataset balance

🧠 Key Insights from EDA

Data shows non-linear boundaries → RBF kernel expected to perform best.

No heavy skew → no major transformation required.

Features have moderate correlations → SVM handles this well.

🤖 Modeling Approach
✔ Baseline SVM Model

Kernel: RBF

Accuracy: ~0.77

AUC: ~0.76

✔ Hyperparameter Tuning (GridSearchCV)

Parameters tuned:

C: [1, 5, 10, 20, 50]

gamma: [‘scale’, 0.1, 0.01]

kernel: ‘rbf’

✔ Best Model Achieved

Kernel: RBF

C: 20

gamma: scale

Test AUC: 0.804

Accuracy: 0.77

✔ Kernel Comparison Summary
Kernel	Accuracy	AUC
Linear	0.72	0.798
Polynomial	0.65	0.724
RBF	0.77	0.843

👉 RBF SVM is the best-performing classifier for this dataset.

📦 Project Structure
SVM-Drug-Response-Classification/
│
├── Drug_Response_Classification.ipynb
├── svm_drug_response_classification_assignment_10.py
│
├── dataset/
│   └── Pharma_Industry.csv
│
├── docs/
│   ├── Support Vector machine.docx
│   └── Drug Response Classification.docx
│
├── images/
│   ├── histogram_grid_v2.jpeg
│   ├── boxplot.jpeg
│   ├── correlation_heatmap.jpeg
│   ├── pairplot.jpeg
│   ├── scatter_plot.jpeg
│   └── count_plot.jpeg
│
├── README.md
└── requirements.txt

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Open the Notebook
Drug_Response_Classification.ipynb

3. Run all cells to reproduce results.
🧩 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🏁 Conclusion

SVM is effective for clinical biomarker-based classification.

The dataset exhibits non-linear patterns, making the RBF kernel optimal.

The final tuned model achieves strong predictive performance, demonstrating applications in precision medicine and drug development research.

📬 Author

Kukkala Kiran Kumar
Machine Learning • Bioinformatics • Computational Biology
GitHub: kirankumar88
