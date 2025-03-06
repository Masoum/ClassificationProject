# Customer churn prediction

## Objective
**The goal is to create a model to predict whether or not a customer will churn.** A thorough analysis will be performed in order to select one model that predict the retaintion of the customer the most accurately. The most accurate model will tested against completely new independent data.


## Data source
We worked with a [dataset](../data/churn.xlsx) containing information from a fictional telco company that provided home phone and internet services to 7043 customers in California in Q3. 

It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index.

More information can be found [here](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

Churn analysis is a method used by organizations to understand the rate at which customers discontinue their business with a company and to identify the reasons and factors contributing to this decision. This type of analysis is crucial in many industries, such as telecommunications, retail, finance, and any subscription-based service.



## Notebooks

You can find all the models' notebooks in [notebook](../notebook/)

---
## Data Cleaning

Data cleaning is crucial for data analysis. The cleaned data can be found here [Cleaned Dataset](data/churn_cleaned.csv)

---
## Analysis

The churn reason were analyzed to pinpoint the factors that pushed a client to leave the company.

![Churn Reason Distribution](../graph/ChurnReason.png)

</br></br>

The distribution of the 'Tenure Months' column indicated that it was going a good predictive feature.
![TenureMonths Distribution](../graph/TenureMonths.png)

</br></br>

The distribution of the 'Contract' column for Churn clients indicated that it was going a good predictive feature.
![TenureMonths Distribution](../graph/Contract.png)

</br></br>
By looking at the correlation between the various columns, we noticed that all the categorical columns that contains a category 'No internet service' is in fact a perfect replica of the columns 'Internet Service' with the category value of 'No'.

Therefore, all categorical columns containing the category 'No internet service' were cleaned, with the exception of the column 'Internet Service'.


![Correlation Heatmap](../graph/Correlation_heatmap_all.png)

</br></br>

---
## Feature Engeneering

Most regression models can only accept numeric values. Therefore, feature engineered is a crucial step to transform categorical columns into numerical columns.

In addition, this is where we will be finding the best predictors of Price for our classification models. Four methods will be use and compared to find the best predictors: The feature engineed csv file can be find in [Feature Engineered Dataset](../data/churn_cleaned_featEng.xlsx)
The best predictors will be used to find the best suitable regression model for the Residential Housing Rental Price accross Canada. 

</br></br>


---
## Our models
---

### **Logistic Regression**

The best parameters for the Logistic Regression model are {'C': 100, 'max_iter': 10000, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.01}.

The features used to build the logistic regression model are the 5 most correlated columns with the Churn Value column.

| Feature                         | Importance |
|--------------------------------|:----------:|
| Tenure Months                 | 0.3522     |
| Internet Service_Fiber optic   | 0.3080     |
| Contract_Two year             | 0.3023     |
| Payment Method_Electronic check | 0.3019     |
| Dependents_Yes                | 0.2485     |


### **K-Nearest Neighbor

In K-Nearest Neighbors (KNN), determining feature importance is not straightforward because KNN is a distance-based algorithm rather than a model that assigns explicit weights to features like decision trees or linear regression.

#### Model Accuracy
| Metric               | Score  |
|----------------------|--------|
| Model accuracy score | 0.7370 |


#### K-Fold Cross Validation - 10 Fold
| Metric                         | Score  |
|--------------------------------|--------|
| Average cross-validation score | 0.7416 |
| Lowest cross-validation score  | 0.7221 |
| Highest cross-validation score | 0.7627 |


#### GridSearchCV Best Parameters
| Hyperparameter         | Best Value       |
|-----------------------|-----------------|
| classifier__n_neighbors | n_neighbors=14, p=2.0 |
| classifier__weights    | uniform         |
| Best score       |        0.7767     |


### **Random Forest**
Understanding which features impact customer churn is crucial for improving retention strategies. The table below displays the **feature importance scores** from a trained **Random Forest model**. These scores indicate how much each feature contributes to predicting whether a customer will churn.

 **Tenure Months (0.168)** is the most important factor, meaning customers who have been with the company longer are less likely to churn.
 **Monthly Charges (0.143)** is also significant, so pricing plays a key role in retention.
 Features like **Internet Service Type, Contract Type** also contribute to churn prediction.
 **Demographic factors like gender and senior citizenship** play a smaller role.

### **Feature Importance for Churn Prediction**
| **Feature**                                  | **Importance Score** |
|---------------------------------------------|--------------------|
| Tenure Months                               | 0.168757          |
| Monthly Charges                             | 0.143493          |
| Latitude                                    | 0.121566          |
| Longitude                                   | 0.119230          |
| Internet Service_Fiber optic                | 0.050034          |
| Contract_Two year                           | 0.041809          |
| Payment Method_Electronic check             | 0.035338          |
| Dependents_Yes                              | 0.034666          |
| Contract_One year                           | 0.026356          |
| Gender_Male                                 | 0.023360          |
| Partner_Yes                                 | 0.022789          |
| Paperless Billing_Yes                       | 0.021448          |
| Online Security_Yes                         | 0.021020          |
| Tech Support_Yes                            | 0.020890          |
| Online Backup_Yes                           | 0.019128          |
| Senior Citizen_Yes                          | 0.018984          |
| Multiple Lines_Yes                          | 0.017637          |
| Device Protection_Yes                       | 0.017438          |
| Streaming Movies_Yes                        | 0.016733          |
| Streaming TV_Yes                            | 0.015196          |
| Internet Service_No                         | 0.014582          |
| Payment Method_Mailed check                 | 0.011278          |
| Payment Method_Credit card (automatic)      | 0.011104          |
| Phone Service_Yes                           | 0.007164          |


### **DecisionTreeClassifier**

The Decision Tree Classifier was chosen to address the unbalanced dataset that we have.

Various scoring methods and weights were tested to address the issue of the unbalanced dataset. However, the most suited scoring method that was found during the process is the 'balanced_accuracy'. The best weights found was {0:0.5, 1:1.5}.

The decision tree was build with 24 features. However, the most important features are :

| Feature                         | Importance |
|--------------------------------|:----------:|
| Contract_Two year            | 0.354717   |
| Contract_One year            | 0.209229   |
| Tenure Months                | 0.099239   |
| Dependents_Yes               | 0.093631   |
| Internet Service_Fiber optic  | 0.092869   |
| Latitude                      | 0.035499   |
| Streaming Movies_Yes         | 0.033793   |
| Monthly Charges              | 0.027313   |
| Longitude                    | 0.014449   |
| Internet Service_No          | 0.013449   |
| Phone Service_Yes            | 0.012041   |
| Payment Method_Electronic check | 0.006196   |
| Online Security_Yes          | 0.003894   |
| Senior Citizen_Yes           | 0.003683   |

In addition, the final decision tree classifier had a depth of 6.

</br></br>

---
# Scores of the various classification models
---

**Explanation of Metrics:**

>    **Precision**: The proportion of true positive predictions out of all positive predictions.</br>
     **Recall**: The proportion of true positive predictions out of all actual positive instances.</br>
     **F1-score**: The harmonic mean of precision and recall, balancing both metrics.</br>
     **Support**: The number of actual occurrence</br>
     

## Results

**Logistic Regression**
|               | Precision | Recall | F1-Score | Support |
|---------------|:---------:|:------:|:-------:|:-------:|
| **Class 0**   | 0.84     | 0.91   | 0.87    | 783     |
| **Class 1**   | 0.65     | 0.49   | 0.56    | 274     |
| **Accuracy**  | -        | -      | 0.80    | 1057    |
| **Macro Avg** | 0.74     | 0.70   | 0.71    | 1057    |
| **Weighted Avg** | 0.79     | 0.80   | 0.79    | 1057    |

</br></br>
**K-Nearest Neighbor** (weights=uniform)

|               | Precision | Recall | F1-Score | Support |
|---------------|:---------:|:------:|:-------:|:-------:|
| **Class 0**   | 0.83     | 0.81   | 0.82    | 783     |
| **Class 1**   | 0.49     | 0.51   | 0.50    | 274     |
| **Accuracy**  | -        | 0.74   | -       | 1057    |
| **Macro Avg** | 0.66     | 0.66   | 0.66    | 1057    |
| **Weighted Avg** | 0.74     | 0.74   | 0.74    | 1057    |

</br></br>
**Random Forest** (scoring=accuracy, class_weights={0: 1, 1: 1})
|               | Precision | Recall | F1-Score | Support |
|---------------|:---------:|:------:|:-------:|:-------:|
| **Class 0**   | 0.86     | 0.90   | 0.88    | 783     |
| **Class 1**   | 0.66     | 0.57   | 0.61    | 274     |
| **Accuracy**  | -        | -      | 0.81    | 1057    |
| **Macro Avg** | 0.76     | 0.73   | 0.74    | 1057    |
| **Weighted Avg** | 0.81     | 0.81   | 0.81    | 1057    |

</br></br>

**DecisionTreeClassifier** (scoring=balanced_accuracy, class_weights={0: 0.5, 1: 1.5})
|               | Precision | Recall | F1-Score | Support |
|---------------|----------|-------|---------|--------|
| Class 0       | 0.93     | 0.67  | 0.78    | 783    |
| Class 1       | 0.48     | 0.86  | 0.61    | 274    |
| Accuracy      | -        | 0.72  | -       | 1057   |
| Macro Avg     | 0.71     | 0.77  | 0.70    | 1057   |
| Weighted Avg  | 0.82     | 0.72  | 0.74    | 1057   |

</br></br>

---
# Validation of best model (Ramdom Forest), of balanced model (DecisionTreeClassifier) and of client's model (Churn Score)
---
In this notebook, we will be validating and comparing different models together

Here are the three models that will be validated and for which we will be comparing the results:
1. The best model found, which is a Random Forest model, based on the accuracy score.
2. The best model based on the balanced accuracy score, which is a Decision Tree Classifier model.
3. The client's actual model from which the Churn Score is calculated.


## Results

The accuracy is not the only scores that matters. As seen during this project, the scoring method best suited to optimize a Classification model is really dependent on the goal that the model need to achieve.

In this case, the client wanted to forecast the hit event the most accurately while minimizing the missed event. Therefore, a scoring method that would have balanced the dataset and try to find the best model that tend to have 0 False Negative, while optimizing the balanced accuracy would have been the best scoring method for this project.

![Confusion Matrix](../graph/tempo.png)
</br></br>

It is a lesson learn that a model with the most accuracy does not always provide the best forecast.

We were not able to provide a better model than the pre-existing model used to produce the Churn Score and suit better the client's needs.


| Balanced Model</br>DecisionTreeClassifier              | Balanced Model</br>Random Forest                  | Client pre-existing model</br> Churn Score               |
|-----------------------|-----------------------|-----------------------|
| ![Confusion Matrix Balanced Model](../graph/ConfusionMatrix_val_BalancedModel.png) | ![Confusion Matrix Best Model](../graph/ConfusionMatrix_val_BestModel1.png) |  ![Confusion Matrix Balanced Model](../graph/ConfusionMatrix_val_ChurnScore.png) | 

</br></br></br>
<center>
    
####  Balanced Model (DecisionTreeClassifier)

</center>

| Metric       | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|--------------|:-------:|:-------:|:--------:|:---------:|:------------:|
| **Precision**| 0.93    | 0.47    | -        | 0.70      | 0.82         |
| **Recall**   | 0.72    | 0.83    | -        | 0.77      | 0.74         |
| **F1-Score** | 0.81    | 0.60    | 0.74     | 0.71      | 0.76         |
| **Support**  | 538     | 166     | 704      | 704       | 704          |


</br></br>
<center>
    
####  Balanced Model (Random Forest)

</center>

| Metric       | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|--------------|:-------:|:-------:|:--------:|:---------:|:------------:|
| **Precision**| 0.86    | 0.66    | -        | 0.76      | 0.81         |
| **Recall**   | 0.90    | 0.57    | -        | 0.73      | 0.81         |
| **F1-Score** | 0.88    | 0.61    | 0.81     | 0.74      | 0.81         |
| **Support**  | 783     | 274     | 1057     | 1057      | 1057         |


</br></br>
<center>
    
####  Client Pre-existing Model (Churn Score)

</center>

| Metric       | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|--------------|:-------:|:-------:|:--------:|:---------:|:------------:|
| **Precision**| 1.00    | 0.37    | -        | 0.69      | 0.85         |
| **Recall**   | 0.49    | 1.00    | -        | 0.74      | 0.61         |
| **F1-Score** | 0.65    | 0.55    | 0.61     | 0.60      | 0.63         |
| **Support**  | 538     | 166     | 704      | 704       | 704          |

</br></br>

---
# Conclusion
---
We learned a valuable lesson.  that a model with the most accuracy does not always provide the best forecast.

We were not able to provide a better model than the pre-existing client's model used to produce the Churn Score.
