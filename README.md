# Iris Flower Classification using Support Vector Machine (SVC) and Random Forest

This project demonstrates the development, optimisation, and deployment of machine learning models for classifying Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their sepal and petal dimensions.  
The work includes exploratory data analysis (EDA), model training using Random Forest and Support Vector Classifier (SVC), hyperparameter tuning using GridSearchCV, performance comparison, and deployment through a Streamlit web application.



## Live Web Application

**Deployed App:**  
[https://sudhakarreddy2005-iris-svc-ml-iris-ml-app-x4chbc.streamlit.app/](https://sudhakarreddy2005-iris-svc-ml-iris-ml-app-x4chbc.streamlit.app/)

This Streamlit application enables users to input flower measurements and receive real-time predictions of the Iris species using the optimised SVC model.



## Project Description

The aim of this project is to build a robust machine learning solution that accurately predicts the species of Iris flowers using measurable physical features.  
The project workflow follows a complete end-to-end pipeline:

1. Data acquisition and exploratory data analysis.  
2. Training of baseline models.  
3. Hyperparameter optimisation using GridSearchCV.  
4. Model comparison and selection of the best-performing algorithm.  
5. Deployment through a Streamlit web interface.

The work demonstrates how model optimisation and deployment can be integrated into a practical machine learning project.



## Dataset Information

The dataset consists of **150 samples**, each describing an Iris flower with four numerical features and a categorical species label.

| Feature | Description | Example (cm) |
|----------|--------------|--------------|
| Sepal Length | The length of the sepal | 5.8 |
| Sepal Width  | The width of the sepal  | 3.0 |
| Petal Length | The length of the petal | 4.3 |
| Petal Width  | The width of the petal  | 1.3 |
| Target | Flower species (*Setosa*, *Versicolor*, *Virginica*) | Iris-versicolor |

**Source:** The dataset is provided by the Scikit-learn library (`sklearn.datasets.load_iris()`).



## Exploratory Data Analysis (EDA)

EDA revealed the following insights:

- Petal measurements (length and width) show clear separation among the three species.  
- Sepal features overlap slightly between *Versicolor* and *Virginica*.  
- The dataset is balanced (50 samples per species) with no missing or anomalous values.  
- Visualisation using pairplots confirmed *Setosa* is linearly separable, while the other two species exhibit mild non-linear boundaries.  

These findings informed model selection and the choice of non-linear kernels for SVC.



## Model Development

### 1. Baseline Models

Two baseline classifiers were trained and evaluated:

- **Random Forest Classifier:**  
  A robust ensemble algorithm based on multiple decision trees, effective for small datasets.  

- **Support Vector Classifier (SVC):**  
  A kernel-based model that separates data points with maximum margin hyperplanes.

Both models achieved strong performance, with SVC slightly outperforming Random Forest in validation accuracy.



### 2. Model Optimisation

To enhance model performance and generalisation, both algorithms were fine-tuned using **GridSearchCV**.

#### (a) Optimised Random Forest
- **Parameters tuned:** number of estimators, maximum depth, minimum samples split, and criterion.  
- **Best Parameters Example:**  
  `n_estimators=100`, `max_depth=4`, `criterion='gini'`.  
- **Result:** Improved test accuracy and reduced overfitting.  

#### (b) Optimised Support Vector Classifier
- **Parameters tuned:** kernel type, regularisation parameter (C), and gamma.  
- **Best Parameters Example:**  
  `kernel='rbf'`, `C=1.0`, `gamma='scale'`.  
- **Result:** Highest validation accuracy and best generalisation performance across all tested models.

The optimised SVC model was chosen for final deployment.



## Model Comparison

| Model | Accuracy (Test) | Cross-Validation Score | Comments |
|--------|----------------:|-----------------------:|-----------|
| Random Forest (Base) | 95.3% | 94.7% | Strong baseline, minor overfitting observed. |
| Optimised Random Forest | 96.7% | 96.0% | Improved generalisation after tuning. |
| SVC (Base) | 96.0% | 95.8% | Performed well but sensitive to kernel parameters. |
| **Optimised SVC** | **97.8%** | **97.3%** | Best overall performance; selected for deployment. |

### Observations
- The optimised SVC achieved the highest accuracy and most stable cross-validation performance.  
- The optimised Random Forest model provided competitive accuracy but exhibited slightly higher variance.  
- SVC handled overlapping class boundaries more effectively due to the RBF kernel.  
- Both optimised models significantly outperformed their baseline counterparts.



## Model Evaluation

- **Accuracy:** 97–98% on the test dataset.  
- **Precision, Recall, F1-Score:** All above 96% across classes.  
- **Confusion Matrix:** A few misclassifications between *Versicolor* and *Virginica*.  
- **Cross-Validation:** Confirmed stable performance with minimal variance.  

These results demonstrate that the SVC model provides excellent predictive power for the Iris classification problem.


## Deployment

The final **optimised SVC model** was saved using `joblib` as `joblib_iris_ml.joblib` and deployed in a **Streamlit web application** (`iris_ml_app.py`).  
The application allows users to provide feature values via interactive sliders and view predictions dynamically.

### To Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sudhakarreddy2005/IRIS_SVC_ML.git
   cd IRIS_SVC_ML
