# AI-Based Student Performance Prediction System — Demo

## Features
- Separate student and teacher sign-in views
- Student performance shown only after matching **both roll number and full name**
- Teacher overview and section-wise student risk lists
- Random Forest prediction plus Extra Trees and Gradient Boosting comparison
- 80/20 training/testing split with MAE, RMSE, and R²
- SHAP explanations (clearly labelled global-importance fallback if SHAP fails)
- Personalized support suggestions
- Interactive What-If analysis
- CSV export of section reports

## Run on Windows
Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Demo credentials
- Student: `student` / `student123`
- Teacher: `teacher` / `teacher123`

To find a valid demo student identity, open `data/demo_students.csv` after the first launch and use a row's exact `Roll_Number` and `Student_Name`.

## Data honesty and limitations
This app generates synthetic records in `data/demo_students.csv` on first launch so it can be demonstrated without private institutional data. It does **not** use the Kaggle StudentPerformanceFactors dataset by default, and synthetic metrics are not evidence of real-world accuracy. The Kaggle source data does not contain names, roll numbers, sections, CGPA, or assignment averages; those fields must be collected separately or clearly labelled as demo data.

Login is hard-coded for local demonstration only. Before any real deployment, implement password hashing, server-side role/section authorization, database storage, audit logs, privacy controls, and institution-approved validation. Risk labels are decision-support signals and must not be the sole basis for consequential academic decisions.
