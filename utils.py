import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

def load_data(data_path):
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def eda_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(x='Department', hue='Attrition', data=df)
    plt.title('Department-wise Attrition')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/department_attrition.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title('Salary vs Attrition')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/salary_attrition.png")
    plt.close()

def evaluate_model(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()
