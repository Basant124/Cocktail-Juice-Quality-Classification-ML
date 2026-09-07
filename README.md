# Cocktail-Juice-Quality-Classification-ML
# 🍹 Cocktail Juice Quality Classification Using Machine Learning

## 📌 Project Overview

This project focuses on **predicting the quality of cocktail juice using Machine Learning classification algorithms**.

The dataset contains physicochemical features of cocktail juice, and the goal is to classify the juice into two quality categories:

* **Bad**
* **Good**

Several Machine Learning algorithms are trained and evaluated to determine which model provides the best classification performance.

---

## 🎯 Project Objectives

* Explore and understand the juice quality dataset.
* Perform **Exploratory Data Analysis (EDA)**.
* Check for missing values and outliers.
* Analyze correlations between features.
* Transform the quality score into categorical classes.
* Scale numerical features.
* Train multiple classification models.
* Compare model accuracy and cross-validation performance.
* Use the trained models to predict juice quality for new inputs.

---

## 📂 Dataset

**Dataset:** `Cocktail Juice Quality_Training Dataset.csv`

The dataset contains numerical physicochemical measurements and a `quality` target variable.

The original quality score is converted into two classes:

| Quality Score | Class |
| ------------- | ----- |
| 3–5           | Bad   |
| 6–7           | Good  |

> The exact class boundaries depend on the binning logic used in the notebook.

---

## 🔍 Exploratory Data Analysis

The project performs several EDA steps:

* Dataset preview using `head()`
* Dataset structure using `info()`
* Statistical summary using `describe()`
* Column inspection
* Missing-value detection
* Quality distribution analysis
* Correlation heatmap
* Boxplots for detecting outliers
* Unique-value analysis

### Outlier Detection

The **Interquartile Range (IQR)** method is used to identify potential outliers in the pH feature.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the CSV dataset using Pandas.
2. Check for missing values.
3. Analyze outliers.
4. Convert the numerical `quality` score into categorical labels.
5. Encode the categorical labels using `LabelEncoder`.
6. Separate features (`X`) and target (`y`).
7. Split the dataset into training and testing sets.
8. Apply `StandardScaler` to normalize the feature values.

### Train/Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=24
)
```

---

## 🤖 Machine Learning Models

The project compares several classification algorithms:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Classifier (SVC)**
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree Classifier**
6. **Gaussian Naive Bayes**
7. **XGBoost Classifier**

---

## 📊 Model Evaluation

Each model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* 10-Fold Cross-Validation

The models are then compared using a summary DataFrame containing:

```text
Algorithm
Accuracy Score (%)
Cross Validation Score (%)
```

An accuracy comparison chart is also created to visualize the performance of the different algorithms.

---

## 🔄 Cross-Validation

The project uses **10-fold cross-validation** to obtain a more reliable estimate of model performance.

```python
cross_val_score(model, X_train, y_train, cv=10)
```

This allows the performance of each model to be evaluated across multiple training/validation splits.

---

## 🔮 Prediction

After training, the models can be used to predict the quality of new juice samples.

Example input:

```python
features = np.array([
    [9.5, 7, 0, 1.9, 0.076, 25, 0.99, 3.5, 0.5]
])

prediction = model.predict(features)
```

The model returns the predicted quality class.

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* Statsmodels

### Algorithms

* Logistic Regression
* Random Forest
* SVM
* KNN
* Decision Tree
* Gaussian Naive Bayes
* XGBoost

---

## 📁 Project Structure

```text
Cocktail-Juice-Quality-Classification-ML/
│
├── README.md
│
├── Cocktail Juice Quality_Training Dataset.csv
│
└── Cocktail_Juice_Quality_Classification.ipynb
```

---

## 📈 Results

The project compares the performance of all implemented Machine Learning models and identifies the strongest-performing model based on **test accuracy and cross-validation score**.

The final model can then be used to classify new cocktail juice samples as **Good** or **Bad**.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Cocktail-Juice-Quality-Classification-ML.git
```

### 2. Install the required libraries

```bash
pip install numpy pandas seaborn matplotlib scikit-learn statsmodels xgboost
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the notebook

Open:

```text
Cocktail_Juice_Quality_Classification.ipynb
```

and run the cells sequentially.

---

## 💡 Key Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Outlier Detection
* Feature Scaling
* Label Encoding
* Classification
* Model Comparison
* Cross-Validation
* Performance Evaluation
* Machine Learning Prediction
* Data Visualization

---

## 👩‍💻 Project Type

**Machine Learning | Classification | Data Analysis | Python**

### ⭐ Project Title

**Cocktail Juice Quality Classification Using Machine Learning**
