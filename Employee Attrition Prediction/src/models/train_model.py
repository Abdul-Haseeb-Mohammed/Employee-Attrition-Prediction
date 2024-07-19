from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

"""
Building the model
We will be building 2 different models:
--> Logistic Regression
--> Support Vector Machine(SVM)

--> The reported average includes the macro average which averages the unweighted mean per label, and the weighted average i.e. averaging the support-weighted mean per label.
--> In classification, the class of interest is considered the positive class. Here, the class of interest is 1 i.e. identifying the employees at risk of attrition.

Reading the confusion matrix (clockwise):
--> True Negative (Actual=0, Predicted=0): Model predicts that an employee would not attrite and the employee does not attrite
--> False Positive (Actual=0, Predicted=1): Model predicts that an employee would attrite but the employee does not attrite
--> False Negative (Actual=1, Predicted=0): Model predicts that an employee would not attrite but the employee attrites
--> True Positive (Actual=1, Predicted=1): Model predicts that an employee would attrite and the employee actually attrites
"""

"""
Logistic Regression Model:
    Logistic Regression is a supervised learning algorithm which is used for binary classification problems i.e. where the dependent variable is categorical and has only two possible values. In logistic regression, we use the sigmoid function to calculate the probability of an event y, given some features x as:
        P(y)=1/(1 + exp(-x))
"""

def train_logistic_regression(x_train, y_train):
    lrmodel = LogisticRegression().fit(x_train, y_train)
    return lrmodel

def train_SVM(x_train, y_train):
    SVM = SVC(kernel = 'linear').fit(x_train, y_train)
    return SVM
