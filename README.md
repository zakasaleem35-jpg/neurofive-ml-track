# neurofive-ml-track

Machine Learning track -> Exploratory Data Analysis (EDA) projects.

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

## Files
- `titanic_eda.ipynb` -> the main notebook covering both tasks

## Tools Used
- Python, pandas, numpy, matplotlib, seaborn
- Google Colab / Jupyter Notebook

## Author
**Zaka Saleem** -> [@zakasaleem35-jpg](https://github.com/zakasaleem35-jpg)
