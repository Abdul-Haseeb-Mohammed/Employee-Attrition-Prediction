import pickle
import warnings
warnings.filterwarnings("ignore")
from src.data.make_dataset import load_data, drop_uneccessary_columns_and_preliminary_data_exploration, view_percentage_of_subcategories, view_mean_of_each_numerical_variable
from src.features.build_features import get_dummies,make_features, standardize, train_test_splitter
from src.models.train_model import train_logistic_regression, train_SVM
from src.models.predict_model import make_predictions
from src.visualization.visualize import plot_histograms, plot_barcharts, plot_correlation_heatmap, metrics_score

if __name__ == "__main__":
    # Load the dataset
    data_path = "Employee Attrition Prediction/data/raw/HR_Employee_Attrition.xlsx"
    df = load_data(data_path)

    #Creating numerical columns
    num_cols=['DailyRate','Age','DistanceFromHome','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears',
            'YearsAtCompany','NumCompaniesWorked','HourlyRate',
            'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TrainingTimesLastYear']

    #Creating categorical variables
    cat_cols= ['Attrition','OverTime','BusinessTravel', 'Department','Education', 'EducationField','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance',
            'StockOptionLevel','Gender', 'PerformanceRating', 'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus','RelationshipSatisfaction']
    
    df = drop_uneccessary_columns_and_preliminary_data_exploration(df, num_cols)
    
    plot_histograms(df, num_cols, "Employee Attrition Prediction/reports/figures/histograms.png")
    
    view_percentage_of_subcategories(df, cat_cols)    
    
    plot_barcharts(df, cat_cols, "Employee Attrition Prediction/reports/figures/barcharts.png")
    
    view_mean_of_each_numerical_variable(df, num_cols)
    
    print("Let's check relationship of numerical variables:")
    plot_correlation_heatmap(df, num_cols, "Employee Attrition Prediction/reports/figures/correlation_matrix_heatmap.png")
    
    print("""Model Building - Approach
    1. Prepare data for modeling
    2. Partition the data into train and test set.
    3. Build model on the train data.
    4. Tune the model if required.
    5. Test the data on test set.""")
    
    #creating list of dummy columns
    to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField', 
                          'EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]
    
    df2 = get_dummies(df, to_get_dummies_for)
    
    #mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No':0}
    dict_attrition = {'Yes': 1, 'No': 0}

    df2 = make_features(df2, dict_OverTime, dict_attrition)
    
    x_scaled = standardize(df2.drop(columns = ['Attrition']))
    
    X_train, X_test, y_train, y_test = train_test_splitter(x_scaled, df2['Attrition'])
    
    #Train the Linear regression model
    lrmodel = train_logistic_regression(X_train, y_train)

    # Save the trained model
    with open('Employee Attrition Prediction/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)

    # Show the metrics of models
    print("Coefficients of Logistic Regression:", lrmodel.coef_)
    print("Intercept of Logistic Regression:", lrmodel.intercept_)
    
    #checking the performance on the training data
    y_pred_train_lr = make_predictions(lrmodel,X_train)
    metrics_score(y_train, y_pred_train_lr, "Employee Attrition Prediction/reports/figures/LR_Model_trainset_metrics_report.png")
    
    #checking the performance on the test dataset
    y_pred_test_lr = make_predictions(lrmodel, X_test)
    metrics_score(y_test, y_pred_test_lr, "Employee Attrition Prediction/reports/figures/LR_Model_testset_metrics_report.png")
    
    """
    Observations:
    We are getting an accuracy of around 90% on train and test dataset.
    However, the recall for this model is only around 50% for class 1 on train and 46% on test.
    As the recall is low, this model will not perform well in differentiating out those employees who have a high chance of leaving the company, meaning it will eventually not help in reducing the attrition rate.
    As we can see from the Confusion Matrix, this model fails to identify the majority of employees who are at risk of attrition.
    """
    
    #Train the Support Vector Machine model
    SVM = train_SVM(X_train, y_train)

    # Save the trained model
    with open('Employee Attrition Prediction/models/SVM.pkl', 'wb') as f:
        pickle.dump(SVM, f)

    #checking the performance on the training data
    y_pred_train_SVM = make_predictions(lrmodel,X_train)
    metrics_score(y_train, y_pred_train_SVM, "Employee Attrition Prediction/reports/figures/SVM_trainset_metrics_report.png")
    
    #checking the performance on the test dataset
    y_pred_test_SVM = make_predictions(lrmodel, X_test)
    metrics_score(y_test, y_pred_test_SVM, "Employee Attrition Prediction/reports/figures/SVM_testset_metrics_report.png")
    
    """
    SVM model with rbf linear is not overfitting as the accuracy is around 80% for both train and test dataset
    Recall for the model only around 50% which implies our model will not correctly predict the employees who are on the risk of attrite.
    The precision is quite good and the model will help not five false positive and will save the cost and energy of the organization.
    """
    