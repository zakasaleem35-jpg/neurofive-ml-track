# neurofive-ml-track

Machine Learning track — Exploratory Data Analysis (EDA) and Machine Learning projects.

## Task 1: Titanic Dataset EDA

Exploratory Data Analysis on the Titanic dataset (`train.csv`) using pandas and numpy.

### What's included
- Loading the dataset with `pandas.read_csv()`
- Inspecting the data using `.head()`, `.info()`, `.describe()`, and `.isnull().sum()`
- Identifying rows/columns count, missing values, and categorical vs numerical columns
- A short markdown "data story" summarizing the key findings

### Key Findings
- The dataset has **891 rows** and **12 columns**.
- Three columns have missing values: `Age` (177), `Cabin` (687), and `Embarked` (2).
- **Numerical columns:** PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
- **Categorical columns:** Name, Sex, Ticket, Cabin, Embarked

## Task 2: Data Cleaning & Visualization

Cleaning the dataset and exploring patterns visually using matplotlib and seaborn.

### What's included
- Handling missing values: filled `Age` with median, `Embarked` with mode, and dropped `Cabin` (~77% missing)
- Outlier detection on `Fare` using a boxplot
- Four visualizations: Age histogram, Age-by-Survival boxplot, Survival-by-Gender bar chart, and a correlation heatmap
- A written analysis of which feature most affects survival

### Key Findings
- **`Sex` most strongly affects survival** — most male passengers did not survive (≈468 died vs 109 survived), while most female passengers survived (≈233 survived vs 81 died).
- Among numerical features, `Pclass` has the strongest correlation with survival (-0.34), followed by `Fare` (0.26). `Age` showed almost no correlation.

## Week 2 · Task 1: Classification Model

Built a supervised classification model to predict Titanic passenger survival.

### Approach
- **Encoding:** One-hot encoded categorical features (`Sex`, `Embarked`) using `pd.get_dummies()`, and dropped identifier columns (`PassengerId`, `Name`, `Ticket`) that don't help prediction.
- **Split:** Divided data into 80% training and 20% testing sets (`train_test_split`, `random_state=42`).
- **Model:** Trained a **Logistic Regression** classifier from scikit-learn.
- **Evaluation:** Measured performance using accuracy score and a confusion matrix.

### Results
- **Final Accuracy: 81.01%** (145 correct out of 179 test passengers)
- **Confusion Matrix:** 90 True Negatives, 55 True Positives, 15 False Positives, 19 False Negatives
- The model identifies non-survivors slightly better (~86%) than survivors (~74%).

### Key Takeaway
Logistic Regression provides a solid baseline (~81%) for survival prediction. Gender and passenger class were the strongest predictors, consistent with the EDA findings from Week 1.

## Files
- `titanic_eda.ipynb` — the main notebook covering all tasks

## Tools Used
- Python, pandas, numpy, matplotlib, seaborn, scikit-learn
- Google Colab / Jupyter Notebook

## Author
**Zaka Saleem** — [@zakasaleem35-jpg](https://github.com/zakasaleem35-jpg)
