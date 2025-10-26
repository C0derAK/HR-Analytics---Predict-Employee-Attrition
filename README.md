# HR-Analytics---Predict-Employee-Attrition

## 🎯 Objective
Analyze employee data to identify key attrition factors and predict future resignations.

## 🛠 Tools Used
- **Python** (Pandas, Seaborn, Scikit-Learn)
- **Power BI** for visualization
- **SHAP** for explainable AI

## 🚀 Steps to Reproduce

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/hr-attrition-project.git
cd hr-attrition-project

# 2️⃣ Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (Linux/Mac)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the model training
python src/train_model.py --data hr_data.csv --output artifacts
