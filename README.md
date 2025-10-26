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

## 🧠 Power BI Dashboard Design

The Power BI dashboard provides a clear view of employee attrition patterns and model-driven insights.

### Dashboard Components
- **KPIs:** Total Employees | Attrition Count | Attrition Rate | Avg. Monthly Income  
- **Attrition by Department:** Highlights departments with highest resignations.  
- **Attrition by Salary Band & Age Group:** Understand income-level and age-based turnover trends.  
- **Job Role Breakdown:** Roles most at risk of attrition.  
- **Promotion & Tenure Scatter Plot:** Visualizes how promotion frequency relates to retention.  
- **Feature Importance:** Shows the most impactful factors (based on SHAP analysis).  

### Screenshot Example:
![Attrition Dashboard](dashboard.png)

> 📊 The dashboard integrates both HR insights and ML explainability to support data-driven retention strategies.
