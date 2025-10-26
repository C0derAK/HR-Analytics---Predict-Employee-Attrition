import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import shap, os, joblib, warnings
from utils import load_data, eda_plots, evaluate_model

warnings.filterwarnings("ignore")

def main(data_path, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(data_path)

    # Encode target
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])

    # Drop unnecessary columns
    X = df.drop(columns=['Attrition', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # EDA
    eda_plots(df, output_dir)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    evaluate_model(y_test, y_pred_lr, "LogisticRegression", output_dir)

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_dt = tree.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    evaluate_model(y_test, y_pred_dt, "DecisionTree", output_dir)

    # SHAP Explainability (for Decision Tree)
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values[1], X, show=False)
    shap.save_html(f"{output_dir}/shap_summary.html", shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], X.iloc[0]))

    # Save models
    joblib.dump(log_reg, f"{output_dir}/logistic_model.pkl")
    joblib.dump(tree, f"{output_dir}/decision_tree_model.pkl")

    print(f"\n✅ Logistic Regression Accuracy: {acc_lr:.2f}")
    print(f"✅ Decision Tree Accuracy: {acc_dt:.2f}")
    print(f"\nArtifacts saved in: {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to HR dataset CSV")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    args = parser.parse_args()
    main(args.data, args.output)
