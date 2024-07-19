import pandas as pd

def load_data(data_path):
    df = pd.read_excel(data_path)
    
    return df

def drop_uneccessary_columns_and_preliminary_data_exploration(data,num_cols):
    print("Preview first rows of dataset:",data.head())
    print("Viewing structure of dataset:",data.info())
    print("""There are 2940 observations and 34 columns. All the column have 2940 non-null values
          i.e. there are no missing values in the data.""")
    print("View number of unique values in columns:\n",data.nunique())
    """Observations:
    --> Employee number is an identifier which is unique for each employee and we can drop this column as it would not add any value to our analysis.
    --> Over18 and StandardHours have only 1 unique value. These column will not add any value to our model hence we can drop them.
    --> On the basis of number of unique values in each column and the data description, we can identify the continuous and categorical columns in the data.
    Let's drop the columns mentioned above and define lists for numerical and categorical columns to apply explore them separately."""
    
    data = data.drop(['EmployeeNumber','Over18','StandardHours'],axis=1)
    
    print("View desriptive statistics of dataset:",data[num_cols].describe().T)
    
    """Observations:
    --> Average employee age is around 37 years. It has a high range, from 18 years to 60, indicating good age diversity in the organization.
    --> At least 50% of the employees live within a 7 km radius from the organization. However there are some extreme values, seeing as the maximum value is 29 km.
    --> The average monthly income of an employee is USD 6500. It has a high range of values from 1K-20K USD, which is to be expected for any organization's income distribution.
    There is a big difference between the 3rd quartile value (around USD 8400) and the maximum value (nearly USD 20000),
    showing that the company's highest earners have a disproportionately large income in comparison to the rest of the employees.
    Again, this is fairly common in most organizations.
    --> Average salary hike of an employee is around 15%. At least 50% of employees got a salary hike 14% or less, with the maximum salary hike being 25%.
    --> Average number of years an employee is associated with the company is 7.
    -->On average, the number of years since an employee got a promotion is 2.18. The majority of employees have been promoted since the last year."""
    
    return data

def view_percentage_of_subcategories(data, cat_cols):
    #Printing the % sub categories of each category
    for i in cat_cols:
        print(data[i].value_counts(normalize=True))
        print('*'*40)
        
    """Observations:
    --> The employee attrition rate is 16%.
    --> Around 28% of the employees are working overtime. This number appears to be on the higher side, and might indicate a stressed employee work-life.
    --> 71% of the employees have traveled rarely, while around 19% have to travel frequently.
    --> Around 73% of the employees come from an educational background in the Life Sciences and Medical fields.
    --> Over 65% of employees work in the Research & Development department of the organization.
    --> Nearly 40% of the employees have low (1) or medium-low (2) job satisfaction and environment satisfaction in the organization, indicating that the morale of the company appears to be somewhat low.
    --> Over 30% of the employees show low (1) to medium-low (2) job involvement.
    --> Over 80% of the employees either have none or very less stock options.
    --> In terms of performance ratings, none of the employees have rated lower than 3 (excellent). 
    About 85% of employees have a performance rating equal to 3 (excellent), while the remaining have a rating of 4 (outstanding).
    This could either mean that the majority of employees are top performers, 
    or the more likely scenario is that the organization could be highly lenient with its performance appraisal process."""

def view_mean_of_each_numerical_variable(data, num_cols):
    #Mean of numerical variables grouped by attrition
    print(data.groupby(['Attrition'])[num_cols].mean())
    
    """Observations:
    -->Employees leaving the company have a nearly 30% lower average income and 30% lesser work experience than those who are not.
    These could be the employees looking to explore new options and/or increase their salary with a company switch.
    -->Employees showing attrition also tend to live 16% further from the office than those who are not.
    The longer commute to and from work could mean they have to spend more time/money every day,
    and this could be leading to job dissatisfaction and wanting to leave the organization.
    We have found out what kind of employees are leaving the company more."""