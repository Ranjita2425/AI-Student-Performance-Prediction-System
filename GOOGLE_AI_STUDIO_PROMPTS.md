# Google AI Studio prompts — use in order

## Prompt 1: Plan before coding
Design an AI-Based Student Performance Prediction System with student and teacher portals, role-based access, student identity verification using both roll number and full name, teacher-assigned sections, risk levels, ML training/testing, SHAP, personalized support suggestions, and What-If analysis. First produce architecture, folder structure, data schema, API contracts, and tests. Do not generate all code yet. Explicitly distinguish source dataset fields from institution-only fields.

## Prompt 2: Data and preprocessing
Create a validated data pipeline for previous/current CGPA, attendance, assignment average, internal assessment, backlogs, study hours, sleep, tutoring, previous scores, and exam score. Handle missing data and categorical encoding correctly. Do not claim the Kaggle StudentPerformanceFactors dataset contains roll numbers, names, sections, CGPA, or assignment averages. Label synthetic records as synthetic.

## Prompt 3: ML models and evaluation
Implement Random Forest, Extra Trees, and Gradient Boosting regression models with a reproducible 80/20 train-test split. Report MAE, RMSE, and R² on held-out data. Prevent data leakage and save model, feature list, preprocessing, and evaluation metrics. Add tests.

## Prompt 4: Explainable AI
Add SHAP TreeExplainer explanations for individual predictions. Show feature contribution values, and explain that positive/negative values are relative to the model baseline. Never label global feature importance as SHAP. If SHAP is unavailable, show a clearly labelled fallback.

## Prompt 5: Secure backend
Create a FastAPI backend with secure password hashing, role-based authorization, input validation, and endpoints for login, student identity verification, teacher sections, risk lists, predictions, SHAP, and What-If. Never trust roles or section identifiers sent by the browser without server-side authorization. Use environment variables for secrets.

## Prompt 6: Frontend dashboards
Create a responsive React dashboard with separate student and teacher login pages, section cards, student risk tables, score charts, recommendations, SHAP charts, and What-If controls. Use clear validation, accessible forms, and an understated academic visual design. Label demo data clearly.

## Prompt 7: What-If analysis
Implement a simulator that changes selected inputs and compares baseline vs simulated predictions from the same trained model. Validate ranges and state that the result is a model estimate, not a guarantee.

## Prompt 8: Test permissions
Add tests for student name+roll matching, student access boundaries, teacher section authorization, invalid values, train/test separation, prediction responses, SHAP responses, and What-If calculations.

## Prompt 9: Integration review
Check imports, field names, API contracts, CORS, environment variables, frontend/backend compatibility, and tests. Provide exact commands to run each service. List unfinished items honestly; do not claim production readiness without verified authentication, authorization, database migrations, and tests.
