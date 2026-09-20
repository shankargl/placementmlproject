# 🎓 Student Placement Prediction

A machine learning project that predicts whether a student is likely to be placed based on academic performance, skills, projects, internship experience, and other student-related features.

The project uses **Logistic Regression** for binary classification and provides an interactive **Streamlit web application** for making predictions.

---

## 📌 Project Overview

Placement prediction is a binary classification problem where the model predicts one of two outcomes:

* `0` → Not Placed
* `1` → Placed

The project demonstrates a complete machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Threshold Tuning
   ↓
Streamlit Application
```

---

## 🎯 Objectives

* Analyze factors related to student placement.
* Perform exploratory data analysis on student data.
* Preprocess categorical and numerical features.
* Build a Logistic Regression classification model.
* Handle class imbalance through appropriate evaluation and threshold selection.
* Evaluate the model using multiple classification metrics.
* Deploy the trained model using Streamlit.

---

## 📊 Dataset

The dataset contains **10,000 student records**.

### Features

| Feature                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| `IQ`                     | Student IQ score                              |
| `Prev_Sem_Result`        | Previous semester result                      |
| `CGPA`                   | Student CGPA                                  |
| `Academic_Performance`   | Academic performance score                    |
| `Internship_Experience`  | Whether the student has internship experience |
| `Extra_Curricular_Score` | Extra-curricular activity score               |
| `Communication_Skills`   | Communication skill score                     |
| `Projects_Completed`     | Number of completed projects                  |
| `Placement`              | Target variable                               |

`College_ID` is removed because it is an identifier rather than a predictive feature.

### Target

```text
No  → 0
Yes → 1
```

The dataset contains an imbalanced target distribution, with more students in the `Not Placed` class than the `Placed` class.

---

## 🔍 Exploratory Data Analysis

The project performs EDA to understand the dataset and identify relationships between student characteristics and placement.

The analysis includes:

* Dataset structure
* Missing-value analysis
* Target distribution
* Statistical summary
* Feature distributions
* Box plots
* Correlation analysis
* Feature vs placement analysis

Example questions explored:

* Does CGPA differ between placed and non-placed students?
* How does academic performance relate to placement?
* Is there a relationship between projects and placement?
* How do communication skills vary across placement outcomes?

---

## ⚙️ Data Preprocessing

### 1. Remove identifier

```python
df = df.drop("College_ID", axis=1)
```

### 2. Encode target

```python
df["Placement"] = df["Placement"].map({
    "No": 0,
    "Yes": 1
})
```

### 3. One-hot encode internship experience

```python
df = pd.get_dummies(
    df,
    columns=["Internship_Experience"],
    dtype=int
)
```

This produces:

```text
Internship_Experience_No
Internship_Experience_Yes
```

### 4. Train/Test Split

The dataset is split into training and testing sets using stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 5. Feature Scaling

`StandardScaler` is fitted only on the training data:

```python
scaler.fit_transform(X_train)
```

and then applied to the test data:

```python
scaler.transform(X_test)
```

This prevents data leakage.

---

## 🤖 Machine Learning Model

The project uses:

### Logistic Regression

Logistic Regression is suitable because placement prediction is a **binary classification problem**.

The model predicts the probability that a student belongs to the `Placed` class.

```text
Input Student Data
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Placement Probability
        ↓
Classification
```

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* ROC Curve

### Why not use accuracy alone?

The dataset is imbalanced, so accuracy alone may not accurately represent how well the model identifies placed students.

Therefore, precision, recall, F1-score and ROC-AUC are also considered.

---

## 🎚️ Threshold Tuning

Instead of relying only on the default classification threshold of `0.5`, the project examines different probability thresholds.

Example:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
```

For each threshold, precision, recall and F1-score are compared.

The selected threshold is saved along with the model and used by the Streamlit application.

This allows the application to control how conservative or aggressive the placement prediction is.

---

## 🌐 Streamlit Application

The project includes an interactive web application built using Streamlit.

Users can enter:

* IQ
* Previous semester result
* CGPA
* Academic performance
* Internship experience
* Extra-curricular score
* Communication skills
* Projects completed

The application then displays:

```text
🎉 Likely to be Placed
```

or

```text
❌ Likely Not to be Placed
```

along with the predicted placement probability.

---

## 🖥️ Application Workflow

```text
User enters student details
          ↓
Create input DataFrame
          ↓
Match training feature order
          ↓
Apply saved StandardScaler
          ↓
Load trained Logistic Regression model
          ↓
Calculate placement probability
          ↓
Apply selected threshold
          ↓
Display prediction
```

---

## 📁 Project Structure

```text
Student-Placement-Prediction/
│
├── app.py
│
├── placement.csv
│
├── placement_model.pkl
├── scaler.pkl
├── features.pkl
├── threshold.pkl
│
├── placement.ipynb
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                  | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| `placement.ipynb`     | Data analysis, preprocessing, training and evaluation |
| `app.py`              | Streamlit application                                 |
| `placement.csv`       | Dataset                                               |
| `placement_model.pkl` | Trained Logistic Regression model                     |
| `scaler.pkl`          | Fitted StandardScaler                                 |
| `features.pkl`        | Training feature order                                |
| `threshold.pkl`       | Selected classification threshold                     |
| `requirements.txt`    | Python dependencies                                   |
| `README.md`           | Project documentation                                 |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Student-Placement-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 📋 Requirements

Create a `requirements.txt` file:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

Possible improvements include:

* Compare Logistic Regression with other classification algorithms.
* Add more student features.
* Improve threshold selection using validation data.
* Add model explainability.
* Add feature importance visualization.
* Deploy the application online.
* Add a prediction history dashboard.
* Experiment with more advanced machine learning models.

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

The prediction should not be treated as a definitive assessment of whether an individual student will actually receive a job offer. Real-world placement decisions depend on many factors that may not be represented in the dataset.

---

## 👨‍💻 Author

**Shankar G L**

Engineering Student | Machine Learning & AI Enthusiast

---

## ⭐ Project Highlights

* 10,000 student records
* Binary classification
* Logistic Regression
* Exploratory Data Analysis
* Feature engineering
* Standardization
* Imbalanced-data evaluation
* Precision/Recall/F1 analysis
* ROC-AUC
* Threshold tuning
* Streamlit deployment
