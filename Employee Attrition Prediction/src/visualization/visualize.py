import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plot_histograms(data, num_cols, save_path=None):
    data[num_cols].hist(figsize=(14,14))
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Histograms plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
    
    """Observations:
    --> The age distribution is close to a normal distribution with the majority of employees between the ages of 25 and 50.
    --> The percentage salary hike is skewed to the right, which means employees are mostly getting lower percentage salary increases.
    --> MonthlyIncome and TotalWorkingYears are skewed to the right, indicating that the majority of workers are in entry / mid-level positions in the organization.
    --> DistanceFromHome also has a right skewed distribution, meaning most employees live close to work but there are a few that live further away.
    --> On average, an employee has worked at 2.5 companies. Most employees have worked at only 1 company.
    --> The YearsAtCompany variable distribution shows a good proportion of workers with 10+ years, indicating a significant number of loyal employees at the organization.
    --> The YearsInCurrentRole distribution has three peaks at 0, 2, and 7. There are a few employees that have even stayed in the same role for 15 years and more.
    --> The YearsSinceLastPromotion variable distribution indicates that some employees have not received a promotion in 10-15 years and are still working in the organization.
    These employees are assumed to be high work-experience employees in upper-management roles, such as co-founders, C-suite employees etc.
    --> The distributions of DailyRate, HourlyRate and MonthlyRate appear to be uniform and do not provide much information.
    It could be that daily rate refers to the income earned per extra day worked while hourly rate could refer to the same concept applied for extra hours worked per day.
    Since these rates tend to be broadly similar for multiple employees in the same department, that explains the uniform distribution they show."""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_barcharts(df, cat_cols, save_path=None):
    # Determine the grid size
    num_plots = len(cat_cols) - 1  # excluding 'Attrition'
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), sharey=True)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    idx = 0
    for col in cat_cols:
        if col != 'Attrition':
            ax = axes[idx]
            cross_tab = pd.crosstab(df[col], df['Attrition'], normalize='index') * 100
            cross_tab.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(col)
            ax.set_ylabel('Percentage Attrition %')
            ax.set_xlabel('')
            ax.legend(loc='upper right')
            idx += 1

    # Remove any empty subplots
    for i in range(idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Barcharts saved to {os.path.abspath(save_path)}")
    else:
        plt.show()


def plot_barcharts1(df, cat_cols, save_path=None):
    for i in cat_cols:
        if i!='Attrition':
            (pd.crosstab(df[i],df['Attrition'],normalize='index')*100).plot(kind='bar',figsize=(8,4),stacked=True)
            plt.ylabel('Percentage Attrition %')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Barcharts saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
        
    """Observations:
    --> Employees working overtime have more than a 30% chance of attrition, which is very high compared to the 10% chance of attrition for employees who do not work extra hours.
    --> As seen earlier, the majority of employees work for the R&D department. The chance of attrition there is ~15%
    --> Employees working as sales representatives have an attrition rate of around 40% while HRs and Technicians have an attrition rate of around 25%. 
    The sales and HR departments have higher attrition rates in comparison to an academic department like Research & Development,
    an observation that makes intuitive sense keeping in mind the differences in those job profiles.
    The high-pressure and incentive-based nature of Sales and Marketing roles may be contributing to their higher attrition rates.
    --> The lower the employee's job involvement, the higher their attrition chances appear to be, with 1-rated JobInvolvement employees attriting at 35%.
    The reason for this could be that employees with lower job involvement might feel left out or 
    less valued and have already started to explore new options,leading to a higher attrition rate.
    --> Employees at a lower job level also attrite more, with 1-rated JobLevel employees showing a nearly 25% chance of attrition.
    These may be young employees who tend to explore more options in the initial stages of their careers.
    --> A low work-life balance rating clearly leads employees to attrite; 30% of those in the 1-rated category show attrition."""

def loss_curve(loss_values,save_path=None):
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss Curve saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
        
def plot_correlation_heatmap(data, num_cols, save_path=None):
    #plotting the correlation between numerical variables
    plt.figure(figsize=(15,8))
    sns.heatmap(data[num_cols].corr(),annot=True, fmt='0.2f', cmap='YlGnBu')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Correlation matrix as a Heatmap saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
        
    """Observations:
    --> Total work experience, monthly income, years at company and years with current manager are highly correlated with each other and with employee age which is easy to understand as these variables show an increase with age for most employees.
    --> Years at company and years in current role are correlated with years since last promotion which means that the company is not giving promotions at the right time.
    
    Now we have explored our data. Let's build the model"""
    
def metrics_score(actual, predicted, save_path=None):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Correlation matrix as a Heatmap saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
    