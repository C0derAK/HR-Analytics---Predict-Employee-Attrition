"""
HR Attrition — Data processing, modeling, and explainability
Run: python train_model.py --data path/to/hr_data.csv --output path/to/artifacts
"""
import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_preprocess(df):
    # Drop duplicates and trivial ID columns
    df = df.drop_duplicates()
    if 'EmployeeID' in df.columns:
        df = df.drop(columns=['EmployeeID'])

    # Convert target to binary
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0}).astype(int)
    else:
        raise ValueError("Dataset must contain 'Attrition' column with values 'Yes'/'No'.")

    # Fill simple numeric missing values with median, categorical with mode
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df, num_cols, cat_cols

def eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Department-wise attrition rate
    if 'Department' in df.columns:
        dept = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
        print("Department-wise attrition rate:\\n", dept)
        dept.plot(kind='bar', title='Attrition rate by Department')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dept_attrition.png'))
        plt.clf()

    # Salary band / MonthlyIncome exploration
    if 'MonthlyIncome' in df.columns:
        df['income_band'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Low','Med-Low','Med-High','High'])
        band = df.groupby('income_band')['Attrition'].mean()
        print("Income band attrition:\\n", band)
        band.plot(kind='bar', title='Attrition by Income Band')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'income_band_attrition.png'))
        plt.clf()

    # Promotions vs Attrition (YearsSinceLastPromotion or YearsAtCompany)
    for col in ['YearsSinceLastPromotion','YearsAtCompany']:
        if col in df.columns:
            df.boxplot(column=col, by='Attrition')
            plt.suptitle('')
            plt.title(f'{col} by Attrition')
            plt.savefig(os.path.join(output_dir, f'{col}_by_attrition.png'))
            plt.clf()

def build_pipeline(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')
    return preprocessor

def train_models(X_train, X_test, y_train, y_test, preprocessor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'dt': DecisionTreeClassifier(max_depth=6, random_state=42)
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'model': pipe, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}
        # Save model
        joblib.dump(pipe, os.path.join(output_dir, f'{name}_pipeline.joblib'))
        # Save confusion matrix image
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cm_{name}.png'))
        plt.clf()
    return results

def explain_model(pipe, X_train, output_dir, feature_names):
    os.makedirs(output_dir, exist_ok=True)
    try:
        import shap
        explainer = shap.Explainer(pipe.named_steps['clf'], pipe.named_steps['pre'].transform(X_train))
        shap_values = explainer(pipe.named_steps['pre'].transform(X_train))
        # Save summary plot
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        plt.clf()
        print("SHAP analysis completed.")
    except Exception as e:
        print("SHAP not available or failed:", e)
        # fallback: permutation importance
        from sklearn.inspection import permutation_importance
        r = permutation_importance(pipe, X_train, pipe.predict(X_train), n_repeats=10, random_state=42, n_jobs=1)
        importances = r.importances_mean
        # Save as CSV
        import pandas as pd
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=False)
        imp_df.to_csv(os.path.join(output_dir, 'feature_importance_permutation.csv'), index=False)
        print("Saved permutation importances.")

def main(args):
    df = load_data(args.data)
    df, num_cols, cat_cols = basic_preprocess(df)
    # Remove target from feature lists
    if 'Attrition' in num_cols: num_cols.remove('Attrition')
    if 'Attrition' in cat_cols: cat_cols.remove('Attrition')
    # Keep a subset of columns for modeling (drop text-heavy columns if present)
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # EDA
    eda(df, args.output)

    # Build preprocessor
    preprocessor = build_pipeline(num_cols, cat_cols)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test, preprocessor, args.output)
    # Print summary
    for name, r in results.items():
        print(f"Model: {name} - acc: {r['acc']:.3f}, prec: {r['prec']:.3f}, rec: {r['rec']:.3f}, f1: {r['f1']:.3f}")

    # Explain top model (choose by f1)
    best_name = max(results.keys(), key=lambda n: results[n]['f1'])
    best_pipe = results[best_name]['model']
    # Derive feature names after preprocessing
    # Note: OneHotEncoder produces many features; this attempt provides combined names where possible
    # Build feature_names safely
    feature_names = []
    try:
        # numeric names
        feature_names += num_cols
        # categorical ones (from onehot)
        ohe = best_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names += cat_names
    except Exception:
        feature_names = [f"f{i}" for i in range(best_pipe.named_steps['pre'].transform(X_train).shape[1])]

    explain_model(best_pipe, X_train, args.output, feature_names)

    # Save evaluation summary
    summary = {name: {'acc': r['acc'], 'prec': r['prec'], 'rec': r['rec'], 'f1': r['f1']} for name, r in results.items()}
    import json
    with open(os.path.join(args.output, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Pipeline complete. Artifacts saved to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to HR CSV file (hr_data.csv)')
    parser.add_argument('--output', required=False, default='artifacts', help='Output folder for artifacts')
    args = parser.parse_args()
    main(args)
