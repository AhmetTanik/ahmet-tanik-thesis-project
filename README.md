<p align="center">
  <samp style="font-size: 18px;">Internship Evaluation & Career Readiness</samp><br/>
  <samp>A mixed-methods research project by <b>Ahmet Tanık</b></samp>
</p>

---
 Academic Overview
 This project investigates how internship experiences influence students’ perceived career readiness by combining both quantitative and qualitative research methods. Real survey data, open-ended reflections, and semi-structured interviews are integrated into a single analysis pipeline to understand which aspects of an internship contribute most to meaningful learning and confidence in entering the job market.

On the quantitative side, the project applies data cleaning, feature engineering, and regression modeling to evaluate how factors such as mentorship quality, task relevance, technical practice, duration, and compensation relate to overall satisfaction and perceived readiness. On the qualitative side, interview transcripts and textual feedback are coded to identify recurring themes around fairness, inclusion, workload, and professional growth.

By linking these two perspectives, the project provides a holistic view of internship quality and offers evidence-based suggestions for universities, students, and employers who want to design internships that are more educationally robust and fair.

Motivation & Research Questions
## 🎯 Motivation

Internships are widely promoted as a bridge between university education and professional work. However, students often report very different experiences: some internships provide strong mentorship and challenging tasks, while others feel repetitive, poorly organized, or even exploitative. Understanding which elements actually help students feel more prepared for their careers is essential for:

- designing better university–industry collaboration,
- improving internship guidelines and expectations,
- and helping students choose internships that truly support their development.

This project aims to provide empirical evidence about what makes an internship valuable from the student perspective.

## ❓ Research Questions

1. Which internship factors (e.g., mentorship, learning opportunities, task relevance, duration, compensation) have the strongest impact on students’ perceived career readiness?
2. How accurately can we model internship satisfaction and readiness using survey-based quantitative metrics?
3. What key themes appear in students’ qualitative reflections about their internships?
4. In what ways do the qualitative insights complement or challenge the quantitative results?
5. How can these findings inform the design of fair and educational internship programmes for future students?
   
Methodology – Akademik Format
## 🔬 Methodology

This thesis follows a mixed-methods design that combines statistical analysis with qualitative interpretation.

### 1. Data Sources

- **Survey data**: structured questionnaire covering supervision quality, task relevance, workload, duration, salary/benefits, and overall satisfaction.
- **Open-ended feedback**: free-text responses where students describe their experience in more detail.
- **Interviews**: semi-structured interviews with selected students to explore expectations, challenges, and perceived outcomes.

### 2. Quantitative Pipeline

1. **Preprocessing**
   - Import raw survey data using `src/preprocess.py`
   - Standardize column names and remove empty rows
   - Encode categorical variables (e.g., paid/unpaid) as numeric features

2. **Feature Engineering**
   - Construct composite indicators for supervision, learning opportunities, and fairness
   - Separate predictors (X) and target variables (y), such as overall satisfaction or readiness score

3. **Modeling**
   - Train regression models using `src/model_architecture.py`
   - Evaluate goodness of fit with R² and RMSE
   - Interpret coefficients to understand factor importance

### 3. Qualitative Pipeline

1. **Transcript preparation**
   - Anonymise and clean interview transcripts and open-ended comments
2. **Coding & theme development**
   - Identify recurring patterns related to learning, support, fairness, and workload
3. **Integration**
   - Compare qualitative themes with quantitative findings to explain why certain factors are more influential than others.

All steps are designed to be reproducible, and the repository structure separates data, code, and analysis notebooks to keep the research workflow clear.

 # ahmet-tanik-thesis-project
### A Research Project on Internship Evaluation & Career Readiness

This repository contains the research work conducted by **Ahmet Tanık** as part of a thesis project analyzing internship experiences using a mixed-methods approach.

## Overview
The project studies 102 internship experiences through surveys, reports, and interviews.

## Methodology
1. Data cleaning
2. Regression modeling
3. Thematic analysis

## Findings
- Strong supervision improves learning.
- Paid internships increase motivation.
- Meaningful tasks outperform repetitive ones.

## Structure
See project folders.

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Load & Inspect the Survey Data

Place your survey file inside the data/ directory.
Example: data/survey.csv

You can load and inspect the data using the preprocessing utilities:
from src.preprocess import load_survey_data, clean_survey_data

df = load_survey_data("data/survey.csv")
df = clean_survey_data(df)

print(df.head())

3. Prepare Features for Modeling

Use the feature engineering helper:
from src.preprocess import prepare_features

X, y = prepare_features(df, target_column="overall_satisfaction")

4. Train the Regression Model

Your project includes a simple regression model architecture:
from src.model_architecture import train_regression_model

model, metrics = train_regression_model(X, y)

print("Training metrics:", metrics)

5. Evaluate the Model

Use the evaluation module:
from src.evaluate import evaluate_model

results = evaluate_model(model, X, y)
print(results)
Example output:
{
  "r2": 0.82,
  "rmse": 0.46
}

6. Utility: Get Project Root Programmatically

Optional, but useful inside notebooks:
from src.utils import project_root

root = project_root()
print(root)

7.Jupyter Notebook Usage
Open:
notebooks/thesis_analysis.ipynb
Inside the notebook, you can:

Explore survey responses

Clean/transform data

Train models

Run qualitative coding analysis

Generate figures for the thesis

8. File Structure Overview
ahmet-tanik-thesis-project/
│
├── data/
│   ├── survey.csv            # Your dataset (not included in repo)
│   └── README.md
│
├── notebooks/
│   └── notes.txt             # Your analysis notes
│
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   ├── model_architecture.py # Regression training
│   ├── evaluate.py           # Model evaluation metrics
│   └── utils.py              # Project helpers
│
├── requirements.txt
└── README.md
Results & Visualizations
## 📊 Results & Visualizations

This repository is designed to support visual exploration of the data in Jupyter notebooks. Typical plots include:

- correlation matrix between key internship factors,
- distribution of satisfaction and readiness scores,
- regression model diagnostics (residuals),
- simple feature-importance style views using model coefficients.

### Example: Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix of internship factors")
plt.tight_layout()
plt.show()
Save the figure as assets/correlation_matrix.png and reference it in the README:
![Correlation matrix](assets/correlation_matrix.png)

Example: Regression Coefficients as “Feature Importance”
import pandas as pd
import matplotlib.pyplot as plt

coefs = pd.Series(model.coef_, index=X.columns).sort_values()

plt.figure(figsize=(8, 6))
coefs.plot(kind="barh")
plt.title("Regression coefficients (standardised features)")
plt.xlabel("Coefficient value")
plt.tight_layout()
plt.show()

Save as assets/feature_importance.png and include:
![Feature importance](assets/feature_importance.png)
 Limitations & Future Work  

```md
6 Limitations & Future Work

Although this project provides useful insights, it also has several limitations:

- **Sample size & selection**: the number of internships analysed is limited and may not represent all disciplines or institutions.
- **Self-reported measures**: satisfaction and readiness are based on students’ self-perception, which can be influenced by mood, expectations, or social desirability.
- **Context-specific factors**: internship regulations, labor laws, and institutional policies differ between countries and universities, which may affect generalisability.
- **Model complexity**: the regression models are intentionally kept simple and interpretable rather than optimised for maximum predictive accuracy.

**Future work** could include:

- collecting a larger and more diverse dataset across multiple institutions,
- incorporating additional features such as company size, sector, or remote/on-site format,
- experimenting with more advanced models (e.g. tree-based methods) for comparison,
- developing an interactive dashboard for universities to monitor internship quality indicators in real time.

