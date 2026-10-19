import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "demo_students.csv")
os.makedirs(os.path.dirname(DATA), exist_ok=True)
FEATURES = ["Previous_CGPA","Current_CGPA","Attendance","Assignment_Average","Internal_Assessment","Backlogs","Hours_Studied","Sleep_Hours","Tutoring_Sessions","Previous_Scores"]

def make_data(n=1200, seed=42):
    rng=np.random.default_rng(seed); rows=[]
    first=["Aarav","Aditi","Ananya","Arjun","Diya","Ishaan","Kavya","Meera","Nikhil","Pranav","Riya","Sahana","Tanvi","Varun","Yash","Zoya"]
    last=["Reddy","Sharma","Patil","Rao","Kumar","Shetty","Gupta","Das","Naik","Joshi","Singh","Poojari"]
    sections=["CSE-A","CSE-B","CSE-C","CSE-D"]
    for i in range(n):
        prev=float(np.clip(rng.normal(7.1,1.15),4,10)); cur=float(np.clip(prev+rng.normal(.15,.65),4,10))
        att=int(np.clip(rng.normal(79,12),35,100)); ass=float(np.clip(rng.normal(73+(cur-7)*2,13),20,100))
        internal=float(np.clip(rng.normal(70+(cur-7)*2,14),15,100)); back=int(np.clip(rng.poisson(max(.15,(7.5-cur)*.45)),0,6))
        hours=float(np.clip(rng.normal(2.5+(cur-6)*.25,.9),.2,7)); sleep=float(np.clip(rng.normal(7,1.1),3,10))
        tutoring=int(np.clip(rng.poisson(1.2),0,6)); prevscore=float(np.clip(rng.normal(68+(prev-7)*4,10),30,100))
        score=float(np.clip(cur*3.4+prev*1.1+att*.105+ass*.105+internal*.09+prevscore*.08+hours*1.1+sleep*.35+tutoring*.65-back*2.4+rng.normal(0,4.8)+4,20,100))
       
        python_score = float(np.clip(rng.normal(ass, 15), 10, 100))
        dbms_score = float(np.clip(rng.normal(ass, 15), 10, 100))
        dsa_score = float(np.clip(rng.normal(ass, 15), 10, 100))
        networks_score = float(np.clip(rng.normal(ass, 15), 10, 100))
        ai_ml_score = float(np.clip(rng.normal(ass, 15), 10, 100))

        rows.append({
            "Roll_Number": f"PU{2024+i%2}{i+1:04d}",
            "Student_Name": f"{first[i%len(first)]} {last[(i*7)%len(last)]}",
            "Section": sections[int(rng.integers(0,4))],
            "Previous_CGPA": round(prev, 2),
            "Current_CGPA": round(cur, 2),
            "Attendance": att,
            "Assignment_Average": round(ass, 1),
            "Python_Programming": round(python_score, 1),
            "DBMS": round(dbms_score, 1),
            "Data_Structures": round(dsa_score, 1),
            "Computer_Networks": round(networks_score, 1),
            "AI_ML": round(ai_ml_score, 1),
            "Internal_Assessment": round(internal, 1),
            "Backlogs": back,
            "Hours_Studied": round(hours, 1),
            "Sleep_Hours": round(sleep, 1),
            "Tutoring_Sessions": tutoring,
            "Previous_Scores": round(prevscore, 1),
            "Exam_Score": round(score, 1)
        })

    return pd.DataFrame(rows)

@st.cache_data
def data_load():
    if not os.path.exists(DATA): make_data().to_csv(DATA,index=False)
    return pd.read_csv(DATA)

@st.cache_resource
def train(df):
    X=df[FEATURES]; y=df["Exam_Score"]
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
    ms={"Random Forest":RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Extra Trees":ExtraTreesRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,random_state=42)}
    metrics=[] 
    for name,m in ms.items():
        m.fit(xt,yt); p=m.predict(xv)
        metrics.append({"Model":name,"MAE":round(mean_absolute_error(yv,p),2),"RMSE":round(mean_squared_error(yv,p)**.5,2),"R²":round(r2_score(yv,p),3)})
    return ms,pd.DataFrame(metrics),xv

def risk(score,att,cgpa,back):
    if score<55 or att<60 or cgpa<5.5 or back>=3: return "High Risk"
    if score<70 or att<75 or cgpa<6.5 or back>=1: return "Medium Risk"
    return "Low Risk"

def advice(r,score):
    a=[]
    if r.Attendance<75:a.append("Create an attendance recovery plan and review missed classes.")
    if r.Current_CGPA<6.5:a.append("Set weekly goals for core subjects and meet an academic mentor.")
    if r.Assignment_Average<60:a.append("Break assignments into milestones and submit drafts early.")
    if r.Internal_Assessment<60:a.append("Schedule targeted revision and practice assessment questions.")
    if r.Backlogs>0:a.append(f"Prioritize clearing {int(r.Backlogs)} backlog(s) with a weekly study plan.")
    if r.Hours_Studied<2:a.append("Try a consistent daily study block.")
    if not a:a.append("Maintain current habits and review progress at the next assessment.")
    return a

st.set_page_config(page_title="Student Performance Intelligence",page_icon="🎓",layout="wide")
st.title("🎓 AI-Based Student Performance Prediction System")
st.caption("Prediction • Risk Analysis • Explainable AI • Personalized Support • What-If Analysis")
df=data_load(); models,metrics,x_test=train(df); model=models["Random Forest"]
if "logged" not in st.session_state: st.session_state.logged=False
if not st.session_state.logged:
    st.subheader("Sign in")
    role=st.radio("Portal",["Student","Teacher"],horizontal=True)
    if role=="Student":
        u=st.text_input("Student username"); p=st.text_input("Password",type="password")
        st.info("Demo credentials: student / student123")
        if st.button("Sign in"):
            if (u,p)==("student","student123"): st.session_state.logged=True; st.session_state.role=role; st.rerun()
            else: st.error("Invalid demo credentials.")
    else:
        u=st.text_input("Teacher username"); p=st.text_input("Password",type="password")
        st.info("Demo credentials: teacher / teacher123")
        if st.button("Sign in"):
            if u.strip().lower() == "teacher" and p.strip() == "teacher123":
                st.session_state.logged = True
                st.session_state.role = "Teacher"
                st.rerun()
            else: st.error("Invalid demo credentials.")
    st.caption("Synthetic demo data is used. Do not treat it as real student records.")
    st.stop()
with st.sidebar:
    st.write("Portal:",st.session_state.role)
    if st.button("Log out"): st.session_state.logged=False; st.rerun()

    pages=["Teacher Overview","Section Explorer"] if st.session_state.role=="Teacher" else ["My Performance","What-If Analysis","Explainable AI"]

    page=st.radio("Navigation",pages)

if st.session_state.role=="Student":
    st.subheader("Student identity verification")
    roll=st.text_input("Roll number"); name=st.text_input("Full name")
    if not roll or not name: st.info("Enter both roll number and full name to continue."); st.stop()
    found=df[(df.Roll_Number.str.casefold()==roll.strip().casefold())&(df.Student_Name.str.casefold()==name.strip().casefold())]
    if found.empty: st.error("No matching record. Check both fields."); st.stop()
    s=found.iloc[0]; row=s[FEATURES].to_frame().T; pred=float(model.predict(row)[0]); level=risk(pred,s.Attendance,s.Current_CGPA,s.Backlogs)
    if page=="My Performance":
        c=st.columns(4); c[0].metric("Predicted score",f"{pred:.1f}/100"); c[1].metric("Risk",level); c[2].metric("Current CGPA",f"{s.Current_CGPA:.2f}/10"); c[3].metric("Attendance",f"{s.Attendance}%")
        st.subheader("Personalized support")
        for x in advice(s,pred): st.write("• "+x)
        st.dataframe(s[["Roll_Number","Student_Name","Section","Previous_CGPA","Current_CGPA","Attendance","Assignment_Average","Internal_Assessment","Backlogs"]].to_frame("Value"))
        
        st.subheader("📚 Subject-Wise Assignment Analysis")

        subject_columns = {
            "Python Programming": "Python_Programming",
            "DBMS": "DBMS",
            "Data Structures": "Data_Structures",
            "Computer Networks": "Computer_Networks",
            "Artificial Intelligence & ML": "AI_ML"
        }

        recommendations = {
            "Python Programming": "Practice Python syntax, functions, and coding problems.",
            "DBMS": "Revise SQL queries, normalization, and database design.",
            "Data Structures": "Practice arrays, linked lists, stacks, queues, and trees.",
            "Computer Networks": "Revise OSI layers, TCP/IP, routing, and protocols.",
            "Artificial Intelligence & ML": "Review ML algorithms, model evaluation, and basic AI concepts."
        }

        subject_rows = []
        for subject, column in subject_columns.items():
            score = float(s[column])
            if score < 50:
                status = "Needs Improvement"
            elif score < 70:
                status = "Moderate"
            else:
                status = "Good"

            subject_rows.append({
                "Subject": subject,
                "Assignment Score (%)": round(score, 1),
                "Status": status,
                "Recommendation": recommendations[subject]
                if score < 70 else "Keep practising to maintain your performance."
            })

        subject_report = pd.DataFrame(subject_rows)
        st.dataframe(subject_report, use_container_width=True, hide_index=True)

        weak_subjects = subject_report[
            subject_report["Assignment Score (%)"] < 50
        ]

        if not weak_subjects.empty:
            st.warning("Prioritize these subjects for improvement:")
            for _, item in weak_subjects.iterrows():
                st.write(
                    f"**{item['Subject']} ({item['Assignment Score (%)']}%)** — "
                    f"{item['Recommendation']}"
                )
        else:
            st.success(
                "No subject has an assignment score below 50%. "
                "Continue working on subjects marked Moderate."
            )

    elif page=="What-If Analysis":
        st.subheader("What-If simulation")
        hours=st.slider("Daily study hours",0.0,8.0,float(s.Hours_Studied),.5); att=st.slider("Attendance (%)",30,100,int(s.Attendance))
        ass=st.slider("Assignment average",0,100,int(s.Assignment_Average)); cg=st.slider("Current CGPA",0.0,10.0,float(s.Current_CGPA),.1)
        hyp=row.copy(); hyp.loc[:,"Hours_Studied"]=hours; hyp.loc[:,"Attendance"]=att; hyp.loc[:,"Assignment_Average"]=ass; hyp.loc[:,"Current_CGPA"]=cg
        new=float(model.predict(hyp)[0]); c=st.columns(3); c[0].metric("Current estimate",f"{pred:.1f}"); c[1].metric("What-If estimate",f"{new:.1f}"); c[2].metric("Change",f"{new-pred:+.1f} points")
        st.caption("This is a model simulation, not a guarantee of future marks.")
    else:
        st.subheader("Explainable AI")
        try:
            import shap
            explainer=shap.TreeExplainer(model); vals=np.array(explainer.shap_values(row)).reshape(-1)
            exp=pd.DataFrame({"Feature":FEATURES,"SHAP contribution":vals}); exp["abs"]=exp["SHAP contribution"].abs(); exp=exp.sort_values("abs",ascending=False)
            st.caption("Positive SHAP values push this prediction above the model baseline; negative values push it below.")
            fig,ax=plt.subplots(figsize=(8,4)); plot=exp.head(8).sort_values("SHAP contribution"); ax.barh(plot.Feature,plot["SHAP contribution"]); ax.axvline(0); ax.set_xlabel("SHAP contribution"); st.pyplot(fig); st.dataframe(exp.drop(columns="abs"))
        except Exception as e:
            st.warning("SHAP could not be loaded. Showing global feature importance instead; this is not a per-student SHAP explanation.")
            st.bar_chart(pd.Series(model.feature_importances_,index=FEATURES).sort_values())
else:
    scored=df.copy(); scored["Predicted_Score"]=model.predict(scored[FEATURES]); scored["Risk_Level"]=scored.apply(lambda r:risk(r.Predicted_Score,r.Attendance,r.Current_CGPA,r.Backlogs),axis=1)
    if page=="Teacher Overview":
        c=st.columns(4); c[0].metric("Students",len(scored)); c[1].metric("Average predicted score",f"{scored.Predicted_Score.mean():.1f}"); c[2].metric("High risk",int((scored.Risk_Level=="High Risk").sum())); c[3].metric("Sections",scored.Section.nunique())
        st.subheader("Risk distribution"); st.bar_chart(scored.Risk_Level.value_counts())
        st.subheader("Section summary"); st.dataframe(scored.groupby("Section").agg(Students=("Roll_Number","count"),Average_CGPA=("Current_CGPA","mean"),Average_Attendance=("Attendance","mean"),Average_Predicted_Score=("Predicted_Score","mean")).round(2))
    elif page=="Section Explorer":
        sec=st.selectbox("Select section",sorted(scored.Section.unique())); levels=st.multiselect("Risk levels",["High Risk","Medium Risk","Low Risk"],default=["High Risk","Medium Risk","Low Risk"])
        view=scored[(scored.Section==sec)&(scored.Risk_Level.isin(levels))].sort_values("Predicted_Score")
        st.write(f"{sec}: {len(view)} students"); st.dataframe(view[["Roll_Number","Student_Name","Current_CGPA","Attendance","Assignment_Average","Backlogs","Predicted_Score","Risk_Level"]])
        
        st.subheader("📚 Subject-Wise Improvement Tracker")

        subject_columns = {
            "Python Programming": "Python_Programming",
            "DBMS": "DBMS",
            "Data Structures": "Data_Structures",
            "Computer Networks": "Computer_Networks",
            "Artificial Intelligence & ML": "AI_ML"
        }

        selected_subject = st.selectbox(
            "Select subject to review",
            list(subject_columns.keys())
        )
        
        selected_risks = st.multiselect(
            "Filter by student risk level",
            ["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )

        subject_col = subject_columns[selected_subject]
        
        subject_view = view[view["Risk_Level"].isin(selected_risks)].copy()

        subject_view["Subject_Assignment_Score"] = subject_view[subject_col]

        subject_view["Subject_Status"] = subject_view[
            "Subject_Assignment_Score"
        ].apply(
            lambda score: "Needs Improvement" if score < 50
            else "Moderate" if score < 70
            else "Good"
        )

        st.write(f"Students in {selected_subject} needing improvement")

        needs_help = subject_view[
            subject_view["Subject_Assignment_Score"] < 50
        ].sort_values("Subject_Assignment_Score")

        if needs_help.empty:
            st.success("No students in this section scored below 50 in this subject.")
        else:
            st.dataframe(
                needs_help[[
                    "Roll_Number", "Student_Name", "Section",
                    subject_col, "Subject_Status", "Risk_Level"
                ]].rename(columns={
                    subject_col: "Assignment Score (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.metric("Students needing subject support", len(needs_help))

        st.download_button("Download section report CSV",view.to_csv(index=False).encode(),f"{sec.lower()}_risk_report.csv","text/csv")
   
