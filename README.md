# AI-Based Student Performance Prediction System

A machine-learning demonstration application that predicts student exam performance, classifies academic risk, and provides personalized academic support suggestions. It includes separate student and teacher portals, What-If analysis, and an Explainable AI page.

> **Demo note:** This version uses a local CSV dataset and demo login credentials. Predictions and sample records are for demonstration and must not be treated as official academic results or real student records.

## Features

### Student portal
- Login using roll number and full name.
- Predicted exam score and academic risk classification.
- Performance details such as CGPA, attendance, assignments, internal assessment, and backlogs.
- Subject-wise assignment analysis for Python Programming, DBMS, Data Structures, Computer Networks, and AI/ML.
- Personalized academic suggestions.
- What-If analysis for study hours, attendance, assignment average, and current CGPA.
- Explainable AI using SHAP when available; otherwise, the app shows global model feature importance.

### Teacher portal
- Teacher login.
- Student count, average predicted score, high-risk count, and section count.
- Risk distribution and section-level summary.
- Model evaluation metrics.
- Section Explorer with section and risk filters.
- Subject-wise improvement tracker.
- CSV download for the filtered section report.

## Technology stack
- **Language:** Python
- **Interface:** Streamlit
- **Data handling:** Pandas, NumPy
- **Machine learning:** scikit-learn
- **Charts:** Matplotlib and Streamlit charts
- **Explainability:** SHAP (optional at runtime)

## Project structure

```text
AI-Based-Student-Performance-System-Demo/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── demo_students.csv
└── production/
    └── (separate backend work; not required to run the Streamlit demo)
```

The app expects `data/demo_students.csv`. If the file is missing, the app generates synthetic demo data. If a CSV already exists, it must include the columns used by the application, including `Roll_Number`, `Student_Name`, `Section`, model features, `Exam_Score`, and subject-score columns.

## Setup and run

### 1. Open the project folder
Open a terminal in the project root—the folder containing `app.py`.

### 2. Create and activate a virtual environment (optional if one already exists)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you already use a project virtual environment, activate that environment instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If `requirements.txt` is missing or incomplete, install the main dependencies:

```powershell
python -m pip install streamlit pandas numpy scikit-learn matplotlib
```

For SHAP explanations, install SHAP too:

```powershell
python -m pip install shap
```

### 4. Start the app

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Demo login credentials

### Student portal
Use a roll number and full name that exist in `data/demo_students.csv`.

Example:
- **Roll number:** `PU20240001`
- **Full name:** `Aarav Reddy`

If the CSV has changed, use a matching record from the current CSV.

### Teacher portal
- **Username:** `teacher`
- **Password:** `teacher123`

These are demo credentials only and are not suitable for a publicly deployed application.

## Machine-learning workflow

1. Load student records from the CSV.
2. Select the input features used for training.
3. Split the dataset into training and test sets.
4. Train Random Forest, Extra Trees, and Gradient Boosting regression models.
5. Evaluate models using MAE, RMSE, and R².
6. Use the Random Forest model to estimate exam scores.
7. Apply rule-based thresholds to assign academic risk levels.
8. Show student insights, What-If simulations, and teacher summaries.

### Model input features
- Previous CGPA
- Current CGPA
- Attendance
- Assignment average
- Internal assessment
- Number of backlogs
- Daily study hours
- Sleep hours
- Tutoring sessions
- Previous scores

### Risk classification
The current demo uses fixed rules:
- **High Risk:** predicted score below 55, attendance below 60%, CGPA below 5.5, or at least 3 backlogs.
- **Medium Risk:** if not High Risk, predicted score below 70, attendance below 75%, CGPA below 6.5, or at least 1 backlog.
- **Low Risk:** none of the above conditions apply.

These thresholds are demonstration rules and should be validated with educators before any real-world use.

## Dataset

The demo dataset is stored at `data/demo_students.csv`. It includes student identifiers, section, academic indicators, subject scores, and exam score.

Generated records are synthetic. If real student data is used, obtain authorization, minimize identifying information, protect access, and follow institutional data-protection requirements.

## Current scope and limitations

- This is a demonstration application, not a validated academic decision system.
- Predictions depend on the dataset and training procedure; they do not guarantee future marks.
- What-If results are model simulations, not causal estimates.
- Risk labels are based on fixed thresholds and should not be treated as professional or institutional determinations.
- SHAP explanations require SHAP to be installed and compatible with the environment.
- The Streamlit demo reads a local CSV. Do not assume it is connected to the separate `production/` backend or React frontend unless that integration has been implemented and tested.
- Demo login credentials are not suitable for production authentication.

## Troubleshooting

### Streamlit command not found

```powershell
python -m streamlit run app.py
```

If Streamlit is not installed in the active environment:

```powershell
python -m pip install streamlit
```

### Student login says no matching record
Check that the roll number and full name correspond to the same row in `data/demo_students.csv`. Capitalization and extra spaces should not matter, but the values must match an existing record.

### CSV columns are missing
Keep expected column names intact. Compare the CSV headers with the required columns checked in `data_load()` in `app.py`.

### SHAP is unavailable

```powershell
python -m pip install shap
```

Run the command in the same Python environment used to start Streamlit. The app can fall back to global feature importance if SHAP cannot be loaded.

## Future improvements

- Connect portals to a secured database and backend API.
- Replace demo login with secure role-based authentication and password handling.
- Validate models on representative, authorized data and report appropriate performance metrics.
- Add monitoring, audit logging, and privacy controls.
- Review risk thresholds and recommendations with educators before real-world use.

## Disclaimer

This project is intended for academic demonstration and learning. Predictions and risk categories should not be used alone to make high-stakes decisions about students.
