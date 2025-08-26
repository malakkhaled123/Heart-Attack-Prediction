# Heart Attack Prediction Project

##  Project Overview

This project applies the **Data Science Life Cycle** to predict the risk of heart attack based on patient health indicators. Using machine learning, we analyze clinical features (e.g., age, blood pressure, glucose, troponin levels) to classify patients into **positive (at risk)** or **negative (not at risk)** categories.

---

##  Data Source

* Dataset: **Heart Attack Clinical Dataset**
* Format: CSV file (`Heart Attack.csv`)
* Size: 1,319 patient records, 9 features (age, gender, pulse, blood pressure, glucose, kcm, troponin, class).
* Note: This dataset was obtained from  https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data .

---

##  Workflow

The project follows the **full Data Science Life Cycle**:

1. **Data Collection & Import**

   * Loaded dataset using `pandas`

2. **Data Exploration**

   * Used `.head()`, `.describe()`, `.info()`, and `.isnull().sum()`
   * Checked distributions of variables (age, gender, class balance)

3. **Data Cleaning & Preprocessing**

   * Verified no missing values
   * Encoded categorical variables (`class` → 0/1)
   * Identified possible outliers (e.g., abnormally high pulse values)

4. **Data Visualization**

   * Age distribution histogram
   * Gender distribution pie chart
   * Class balance bar chart
   * Confusion matrix heatmap

5. **Feature Processing & Engineering**

   * Train-test split (80/20)
   * Features: age, gender, pulse, blood pressures, glucose, kcm, troponin
   * Target: class (Positive/Negative)

6. **Model Building**

   * Applied **Logistic Regression** for classification
   * Trained model using Scikit-learn

7. **Model Evaluation**

   * Confusion matrix
   * Accuracy, Precision, Recall, F1-score

---

##  Results

* **Accuracy:** 81%
* **Negative Class:** Precision = 0.73, Recall = 0.79, F1 = 0.76
* **Positive Class:** Precision = 0.86, Recall = 0.81, F1 = 0.84
* **Macro Avg F1:** 0.80 (balanced performance across classes)
* **Weighted Avg F1:** 0.81 (slightly favors Positive class, since dataset has more positive cases)

---

##  Conclusions

* The Logistic Regression model achieved good performance (**81% accuracy**).
* It performs **better at predicting Positive (at risk) patients**, which is critical in a medical setting since false negatives (missed risks) can be dangerous.
* There is still room for improvement:

  * Handle outliers and apply feature scaling
  * Compare with other ML models (Random Forest, SVM, etc.)
  * Perform hyperparameter tuning

---

## 🚀 How to Run the Project

1. Clone the repository

   ```bash
   git clone https://github.com/malakkhaled123/Heart-Attack-Prediction
   cd heart-attack-prediction
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook

   ```bash
   jupyter notebook Heart_Attack_Prediction.ipynb
   ```

---

Would you like me to also **write a short `requirements.txt` file** for your repo (listing the needed Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn)?
