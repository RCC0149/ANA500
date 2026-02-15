ANA500_Titanic_Micro-Project

---

# Titanic Survival Prediction — Advanced Feature Engineering & Model Evaluation

**Author:** Randall C. Crawford
**Course:** ANA 500
**Project Type:** Applied Machine Learning Micro-Project
**Date:** June 2, 2025

---

## 📌 Project Overview

This project investigates **which passenger attributes were most predictive of survival** during the Titanic disaster by applying **rigorous feature engineering, historically informed imputation strategies, and comparative modeling** across multiple machine-learning algorithms.

Unlike typical Titanic analyses, this work emphasizes:

* **Data integrity and realism during imputation**
* **Family-level inference**
* **Iterative feature reduction and validation**
* **Model comparison beyond accuracy alone**

The final deliverable includes a trained classifier used to **predict survival outcomes for passengers with missing labels**, along with a detailed analytical workflow documented in a Jupyter Notebook.

---

## 🎯 Problem Statement

> Identify which passenger characteristics (e.g., age, sex, socio-economic class, fare, family structure, title, embarkation port) were most predictive of survival, and build a model capable of predicting survival for unlabeled passengers.

---

## 🧠 Key Contributions & Differentiators

* **Historically informed imputation** (fare, embarkation port, age)
* **Family-based age inference using surnames, tickets, and relationships**
* **Title extraction and consolidation to capture socio-cultural structure**
* **Explicit investigation (and justified removal) of deck-level features**
* **Extensive model benchmarking** (linear, tree-based, ensemble, neural)

This approach goes significantly beyond standard mean/median imputation or naïve feature dropping.

---

## 📊 Dataset Description

* **Source:** Classic Titanic dataset
* **Total Records:** 1,309 passengers
* **Training Set:** 891 passengers (known survival outcome)
* **Test Set:** 418 passengers (survival outcome missing)
* **Target Variable:** `Survived` (0 = No, 1 = Yes)

### Original Features

* Numeric: `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`
* Categorical: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`

---

## 🛠 Feature Engineering & Data Preparation

### Major Feature Engineering Steps

* **Title extraction** from passenger names (Mr, Mrs, Miss, Master, etc.)
* **Surname + Ticket concatenation** to create `FamilyID`
* **FamilySize** derived from `SibSp + Parch`
* **TicketPrefix / TicketNumber parsing**
* **Age categorization** into meaningful life stages
* **Deck extraction** from cabin values (later removed after analysis)

### Imputation Strategy Highlights

* **Fare:** Median fare imputed using *passenger class + embarkation port*
* **Embarked:** Manually imputed using shared ticket/cabin historical research
* **Age:** Inferred using title, family size, siblings/parents, and family ID
* **Deck:** Investigated extensively and **intentionally discarded** due to insufficient evidence for accurate imputation

This decision-making process is fully documented and justified in the notebook .

---

## 📈 Exploratory Data Analysis

* Distributional analysis (histograms, boxplots)
* Normality testing (Anderson-Darling, QQ plots)
* Skewness analysis
* Survival stratification by:

  * Class
  * Sex
  * Age
  * Family structure
  * Title
  * Fare
  * Embarkation port

These steps guided both **feature selection** and **model design**.

---

## 🤖 Models Evaluated

The following models were trained, tuned, and evaluated:

* **Logistic Regression**
  * Backward elimination
  * L1 (Lasso), L2 (Ridge), ElasticNet
* **Random Forest**
  * Feature-importance-driven pruning
* **Support Vector Machines**
  * Linear and RBF kernels
* **Gradient Boosting**
* **AdaBoost**
* **XGBoost**
* **Neural Network (TensorFlow / Keras)**

Hyperparameters were optimized using grid search or empirical tuning.

---

## 🏆 Model Selection & Results

**Final Model Chosen:** **XGBoost Classifier**

### Why XGBoost?

* Best balance of:
  * Accuracy
  * Macro-averaged F1 score
  * Confusion matrix performance
* ROC-AUC was similar across models and therefore not the primary discriminator

### Final Metrics

* **Accuracy:** ~0.844
* **Macro F1-Score:** ~0.84
* **Correct Predictions:** 752
* **Misses:** 139

The selected model was then used to **generate survival predictions for the unlabeled test dataset**, exported as a `.csv` file for downstream use or Kaggle submission .

---

## 📂 Repository Structure

├── ANA_500_Titanic_Micro-Project.ipynb   # Full analysis and modeling workflow
├── titanic.csv                           # Dataset
├── predictions.csv                      # Final test-set predictions
├── ANA500_Titanic_Micro-Project4.pptx   # Project presentation
├── ANA500_Titanic_Micro-Project4.docx   # Written report
└── README.md                            # Project documentation

---

## 🧪 Technologies & Libraries Used

* Python, NumPy, Pandas
* Scikit-learn
* Statsmodels
* XGBoost
* TensorFlow / Keras
* SHAP, Boruta
* Matplotlib, Seaborn

(Complete import list available in the notebook) 

---

## 🚀 Future Work

* Further validation of age imputation logic
* Exploration of survival probability calibration
* Kaggle competition submission
* Explainability extensions using SHAP across ensemble models

---

## 🧾 Acknowledgments

This project was completed as part of **ANA 500**, serving as a strong foundation for advanced coursework in **AI, optimization, and applied machine learning**.

---


