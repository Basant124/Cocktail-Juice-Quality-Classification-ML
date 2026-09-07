# 🍹 Cocktail Juice Quality Classification Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning classification techniques to predict the quality of cocktail juice based on its physicochemical properties.

The original dataset contains a numerical `quality` score. The project transforms this score into two categories:

* **Bad**
* **Good**

Multiple Machine Learning algorithms are trained, evaluated, and compared to identify the best-performing model.

---

## 🎯 Objectives

* Explore and understand the dataset.
* Perform Exploratory Data Analysis (EDA).
* Identify missing values and potential outliers.
* Analyze relationships between numerical features.
* Transform the quality score into categorical classes.
* Encode the target variable.
* Scale numerical features.
* Train multiple classification models.
* Compare model performance using accuracy and cross-validation.
* Predict the quality of new juice samples.

---

## 📂 Dataset

**Dataset:** `Cocktail Juice Quality_Training Dataset.csv`

The dataset contains physicochemical measurements along with a `quality` target variable.

The original quality score is transformed into two classes:

| Quality               | Class |
| --------------------- | ----- |
| Lower quality scores  | Bad   |
| Higher quality scores | Good  |

> The exact classification boundaries are defined in the notebook using `pd.cut()`.

---

## 🔍 Exploratory Data Analysis

Several EDA techniques are applied to understand the dataset:

* Dataset preview using `head()`
* Dataset structure using `info()`
* Statistical analysis using `describe()`
* Column inspection
* Missing-value detection
* Quality distribution analysis
* Correlation analysis
* Correlation heatmap
* Boxplots for outlier detection
* Unique-value analysis

### 📊 Outlier Detection

The **Interquartile Range (IQR)** method is used to identify potential outliers in the `pH` feature.

The calculation is based on:

* Q1 — 25th percentile
* Q3 — 75th percentile
* IQR = Q3 − Q1
* Lower Bound = Q1 − 1.5 × IQR
* Upper Bound = Q3 + 1.5 × IQR

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Inspect the dataset structure.
3. Check for missing values.
4. Analyze potential outliers.
5. Convert numerical quality scores into categorical labels.
6. Encode the target variable using `LabelEncoder`.
7. Separate features (`X`) and target (`y`).
8. Split the dataset into training and testing sets.
9. Standardize the numerical features using `StandardScaler`.

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

The project compares the following classification algorithms:

* Logistic Regression
* Random Forest Classifier
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Gaussian Naive Bayes
* XGBoost Classifier

---

## 📊 Model Evaluation

Each model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* 10-Fold Cross-Validation

The results are collected into a comparison table containing:

| Metric                 | Description                         |
| ---------------------- | ----------------------------------- |
| Accuracy Score         | Performance on the test dataset     |
| Cross-Validation Score | Average performance across 10 folds |

The project also generates visual comparisons of the model accuracy scores.

---

## 🔄 Cross-Validation

To obtain a more reliable estimate of model performance, **10-fold cross-validation** is applied.

```python
cross_val_score(
    model,
    X_train,
    y_train,
    cv=10
)
```

The mean cross-validation score is then calculated for each algorithm.

---

## 🔮 Prediction

After training the models, new juice samples can be passed to the trained classifier to predict their quality.

Example:

```python
features = np.array([
    [9.5, 7, 0, 1.9, 0.076, 25, 0.99, 3.5, 0.5]
])

prediction = model.predict(features)

print("Prediction:", prediction)
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

### Statistical Analysis

* Statsmodels

---

## 📁 Project Structure

```text
Cocktail-Juice-Quality-Classification-ML/
│
├── README.md
├── Cocktail Juice Quality_Training Dataset.csv
│
└── Cocktail_Juice_Quality_Classification.ipynb
```

---

## 📈 Results

The project evaluates and compares seven Machine Learning classification algorithms.

The final model can be selected based on its:

* Test Accuracy
* Cross-Validation Score
* Classification Report
* Confusion Matrix

This comparison helps determine which algorithm performs best for predicting cocktail juice quality.

> Model performance values are generated directly from the notebook and may vary depending on the dataset and model configuration.

---

## 💡 Key Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Outlier Detection
* Data Preprocessing
* Feature Scaling
* Label Encoding
* Machine Learning Classification
* Model Comparison
* Cross-Validation
* Model Evaluation
* Data Visualization
* Predictive Modeling

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Cocktail-Juice-Quality-Classification-ML.git
```

### 2. Install Dependencies

```bash
pip install numpy pandas seaborn matplotlib scikit-learn statsmodels xgboost
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

Open:

```text
Cocktail_Juice_Quality_Classification.ipynb
```

Run the notebook cells sequentially.

---

## 👩‍💻 Project Type

**Machine Learning | Classification | Data Analysis | Python**

---

## ⭐ Project Title

**Cocktail Juice Quality Classification Using Machine Learning**
