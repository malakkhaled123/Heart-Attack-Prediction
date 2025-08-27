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

## Workflow

The project follows the **full Data Science Life Cycle**:

1. **Data Collection**

   * The dataset was imported as a CSV file.

2. **Data Exploration**

   * Explored data using `.head()`, `.info()`, `.describe()`.
   * Visualized distributions of age, gender, and heart attack labels using histograms, bar charts, and pie charts.
   * Checked correlations between features using a heatmap.

3. **Data Cleaning & Preprocessing**

   * Renamed columns for clarity.
   * Handled outliers in the `impulse` column.
   * Checked for missing values.
   * Encoded the target variable (`class`) using LabelEncoder.

4. **Feature Engineering & Scaling**

   * Created a new feature `pulse_pressure = systolic_bp - diastolic_bp`.
   * Scaled features using `StandardScaler`.

5. **Model Building**

   * Split the dataset into training (80%) and test (20%) sets.
   * Applied **Logistic Regression** for classification.

6. **Model Evaluation**

   * Evaluated the model using a confusion matrix, accuracy score, and classification report (precision, recall, F1-score, macro average, weighted average).
   * Interpretation of metrics:

     * The model achieved **83% accuracy**.
     * Performs slightly better at predicting positive cases (patients at risk).
     * Macro average shows balanced performance across classes.
     * Weighted average accounts for class imbalance in the dataset.


---

**Results**

* Accuracy: 83%
* Negative Class: Precision = 0.77, Recall = 0.80, F1 = 0.78
* Positive Class: Precision = 0.87, Recall = 0.84, F1 = 0.86
* Macro Avg F1: 0.82 (balanced performance across classes)
* Weighted Avg F1: 0.83 (slightly favors Positive class, since dataset has more positive cases)

---

**Comparing LR with SVM**
*	Applied Support Vector Machine (SVM) with three kernels: linear, polynomial, and RBF.
	* Results:
      * Linear kernel: Accuracy = ~0.81
      * Polynomial kernel: Accuracy = ~0.70
      * RBF kernel: Accuracy = ~0.76
*	The linear kernel performed best.
*	Compared Logistic Regression (83%) vs. Best SVM (linear kernel, ~81%).
*	Logistic Regression slightly outperformed SVM, though both showed comparable performance.

---

**Conclusions**

*	The Logistic Regression model achieved good performance (83% accuracy).
*	The Logistic Regression model performs better at predicting Positive (at risk) patients, which is critical in a medical setting since false negatives (missed risks) can be dangerous.
*	There is still room for improvement:
    * Using k-fold cross-validation on the training set to choose the best SVM kernel, instead of picking the kernel based only on test accuracy.
    * Perform hyperparameter tuning to optimize model performance

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


