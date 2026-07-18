# neurofive-ml-track

Machine Learning track — Exploratory Data Analysis (EDA) projects.

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

### Files
- `titanic_eda.ipynb` — the main EDA notebook

### Tools Used
- Python, pandas, numpy
- Google Colab / Jupyter Notebook

## Author
**Zaka Saleem** — [@zakasaleem35-jpg](https://github.com/zakasaleem35-jpg)
