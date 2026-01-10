EMPLOYEE ATTRITION PREDICTION SYSTEM
________________________________________
1.	GROUP MEMBERS AND ROLES
________________________________________
Member 1: Suada Terolli
Role: Random Forest Model, Streamlit application, PowerPoint Presentation
Member 2: Alesia Dardha
Role: Rule-based Model, Hybrid Sequential Model, Report
Both members:
Role: Data Preprocessing, Model Comparison, README.txt
________________________________________
2.	PROJECT DESCRIPTION
________________________________________
This project addresses the problem of employee attrition prediction, which is an imbalanced classification task where most employees remain at the company and only a small portion leave.
The goal is to predict employees who are at risk of leaving using:
•	A rule-based system
•	A machine learning model (Random Forest)
•	A hybrid sequential system combining both approaches
The system prioritizes recall to reduce the risk of missing employees who are likely to leave.
________________________________________
3.	AI APPROACH OVERVIEW 
________________________________________
Three AI approaches are implemented and compared:
1.	Rule-Based System
A conservative, interpretable system using expert-defined rules.
Rules are calibrated using training-set quantiles (e.g., low income, high overtime, low experience).
The system may abstain when no rule applies.
2.	Random Forest Model
A supervised machine learning model trained on historical employee data.
It outputs a probability of attrition and uses a custom decision threshold (default 0.30) to improve recall in an imbalanced dataset. The feature importance is also generated.
3.	Hybrid Sequential System
The rule-based system is applied first.
If a rule produces a decision, it is used directly.
If no rule fires, the Random Forest prediction is used.

All models are evaluated on a held-out test set.
________________________________________
4.	PROJECT STRUCTURE
________________________________________
Typical project structure:
project-root/
│
├─ data/
│ └─ HR-Employee-Attrition.csv
│
├─ models/
│ ├─ random_forest.pkl
│ ├─ feature_names.pkl
│ └─ feature_importance.csv
│
├─ src/
│ ├─ app.py
│ ├─ preprocessing.py
│ ├─ random_forest.py
│ ├─ rule_based.py
│ └─ hybrid_sequential.py
│
└─ README.txt
________________________________________
5.	REQUIREMENTS AND INSTALLATION
________________________________________
Python version:
•	Python 3.10 or newer recommended
Required libraries:
•	pandas
•	scikit-learn
•	joblib
•	streamlit
Installation (recommended using a virtual environment):
python -m venv venv
Windows:
venv\Scripts\activate
Linux / macOS:
source venv/bin/activate
Install dependencies:
pip install pandas scikit-learn joblib streamlit
________________________________________
6.	HOW TO RUN THE PROJECT
________________________________________
1.	Train the Random Forest model:
python -m src.random_forest
This generates:
•	models/random_forest.pkl
•	models/feature_names.pkl
•	models/feature_importance.csv
2.	Evaluate the rule-based system:
python -m src.rule_based
3.	Evaluate the hybrid sequential system:
python -m src.hybrid_sequential
4.	Run the Streamlit application:
streamlit run src/app.py
Open the local URL shown in the terminal (usually http://localhost:8501).
________________________________________
7.	EVALUATION METHODOLOGY
________________________________________
•	Data is split into training and test sets using stratified sampling.
•	Random Forest is trained on the training set only.
•	Rule-based thresholds are calibrated using the training data.
•	All performance metrics are reported on the test set.
Reported metrics:
•	ROC-AUC
•	Precision
•	Recall
•	F1-score
•	Rule coverage (percentage of cases where rules apply)
________________________________________
8.	NOTES
________________________________________
•	If the Streamlit app shows outdated results, clear the cache from the Streamlit menu and refresh.
•	Ensure the dataset is placed inside the data/ folder before running the project.
________________________________________
9.	ACADEMIC USE
This project was developed for academic purposes.
All group members contributed to the implementation and analysis.

