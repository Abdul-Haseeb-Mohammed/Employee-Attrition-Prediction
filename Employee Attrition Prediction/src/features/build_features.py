import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

def get_dummies(data, to_get_dummies_for):
     #creating dummy variables
    data = pd.get_dummies(data = data, columns= to_get_dummies_for, drop_first= True)

    return data

def make_features(data, dict_OverTime, dict_attrition):
    data['OverTime'] = data.OverTime.map(dict_OverTime)
    data['Attrition'] = data.Attrition.map(dict_attrition)
    
    return data

def standardize(X):
    """
    The independent variables in this dataset have different scales. When features have different scales from each other, there is a chance that a higher weightage will be given to features that have a higher magnitude, and they will dominate over other features whose magnitude changes may be smaller but whose percentage changes may be just as significant or even larger. This will impact the performance of our machine learning algorithm, and we do not want our algorithm to be biased towards one feature.
    The solution to this issue is Feature Scaling, i.e. scaling the dataset so as to give every transformed variable a comparable scale.
    In this problem, we will use the Standard Scaler method, which centers and scales the dataset using the Z-Score.
    It standardizes features by subtracting the mean and scaling it to have unit variance.
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean of the training samples (zero) and s is the standard deviation of the training samples.
    """
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns)

def train_test_splitter(X_scaled, Y):
    """
    Splitting the data into 80% train and 20% test set
    Some classification problems can exhibit a large imbalance in the distribution of the target classes: 
        for instance there could be several times more negative samples than positive samples. 
        In such cases it is recommended to use the stratified sampling technique to ensure that 
        relative class frequencies are approximately preserved in each train and validation fold.
    """
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.3,random_state=1,stratify=Y)
    
    print(f"Number of samples in the training set: {len(y_train)}")
    print(f"Number of samples in the test set: {len(y_test)}")
    print("Class distribution in the training set:", Counter(y_train))
    print("Class distribution in the test set:", Counter(y_test))
    print(Counter(y_train)[0]/len(y_train))
    print(Counter(y_train)[1]/len(y_train))
    print(Counter(y_test)[0]/len(y_test))
    print(Counter(y_test)[1]/len(y_test))
    
    return x_train,x_test,y_train,y_test