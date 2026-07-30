neurofive-ml-track
Machine Learning track 
Exploratory Data Analysis (EDA) and Machine Learning projects.

Task 1: Titanic Dataset EDA
Exploratory Data Analysis on the Titanic dataset (`train.csv`) using pandas and numpy.
What's included

* Loading the dataset with `pandas.read_csv()`
* Inspecting the data using `.head()`, `.info()`, `.describe()`, and `.isnull().sum()`
* Identifying rows/columns count, missing values, and categorical vs numerical columns
* A short markdown "data story" summarizing the key findings

Key Findings

* The dataset has 891 rows and 12 columns.
* Three columns have missing values: `Age` (177), `Cabin` (687), and `Embarked` (2).
* Numerical columns: PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
* Categorical columns: Name, Sex, Ticket, Cabin, Embarked

Task 2: Data Cleaning & Visualization
Cleaning the dataset and exploring patterns visually using matplotlib and seaborn.
What's included

* Handling missing values: filled `Age` with median, `Embarked` with mode, and dropped `Cabin` (~77% missing)
* Outlier detection on `Fare` using a boxplot
* Four visualizations: Age histogram, Age-by-Survival boxplot, Survival-by-Gender bar chart, and a correlation heatmap
* A written analysis of which feature most affects survival

Key Findings

* `Sex` most strongly affects survival : most male passengers did not survive (≈468 died vs 109 survived), while most female passengers survived (≈233 survived vs 81 died).
* Among numerical features, `Pclass` has the strongest correlation with survival (-0.34), followed by `Fare` (0.26). `Age` showed almost no correlation.

Week 2 · Task 1: Classification Model
Built a supervised classification model to predict Titanic passenger survival.
Approach

* Encoding: One-hot encoded categorical features (`Sex`, `Embarked`) using `pd.get_dummies()`, and dropped identifier columns (`PassengerId`, `Name`, `Ticket`) that don't help prediction.
* Split: Divided data into 80% training and 20% testing sets (`train_test_split`, `random_state=42`).
* Model: Trained a Logistic Regression classifier from scikit-learn.
* Evaluation: Measured performance using accuracy score and a confusion matrix.

Results

* Final Accuracy: 81.01% (145 correct out of 179 test passengers)
* Confusion Matrix: 90 True Negatives, 55 True Positives, 15 False Positives, 19 False Negatives
* The model identifies non-survivors slightly better (~86%) than survivors (~74%).

Key Takeaway
Logistic Regression provides a solid baseline (~81%) for survival prediction. Gender and passenger class were the strongest predictors, consistent with the EDA findings from Week 1.

Week 2 · Task 2: House Price Prediction with Linear Regression
Built a regression model to predict California housing prices.
Approach

* Dataset: California Housing dataset (loaded via `sklearn.datasets.fetch_california_housing`) : 20,640 rows, no missing values
* Feature selection: Checked correlation of all features with `MedHouseVal` using `df.corr()`, then selected the 5 with the strongest relationships: `MedInc`, `AveRooms`, `HouseAge`, `Latitude`, `Longitude`
* Split: Divided data into 80% training and 20% testing sets (`train_test_split`, `random_state=42`) : 16,512 training rows and 4,128 test rows
* Model: Trained a Linear Regression model from scikit-learn
* Evaluation: Measured performance using RMSE and R² score, and visualized results with a predicted-vs-actual scatter plot

Results

* RMSE: 0.74
* R² Score: 0.58
* The scatter plot shows a reasonably tight cluster around the ideal prediction line for lower-to-mid range prices, with more scatter (and a capped value at 5.0) for higher-priced homes, a known artifact of this dataset, where `MedHouseVal` is capped at $500,000

Key Takeaway
Linear Regression achieved an R² score of 0.58, meaning the model explains about 58% of the variation in house prices using just 5 features. In practical terms, the model gives a useful "ballpark" estimate for most homes, but its accuracy drops for higher-value properties,  likely because price at the high end depends on factors (like exact neighborhood desirability) not captured by these 5 features.

## Week 3 · Task 1: Model Evaluation & Tuning

Evaluated the Logistic Regression classifier beyond accuracy and applied hyperparameter tuning to try to improve it.

### Evaluation (Precision, Recall, F1)
Used `classification_report` to look at per-class performance:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Died (0) | 0.83 | 0.86 | 0.84 |
| Survived (1) | 0.79 | 0.74 | 0.76 |

The model has a recall of only **0.74 for survivors**, meaning it misses about 26% of actual survivors. This weakness is invisible in the overall accuracy (81%), which is why accuracy alone can be misleading for imbalanced datasets.

### Hyperparameter Tuning
Tuned two hyperparameters (`C` and `solver`) using both GridSearchCV and RandomizedSearchCV:

| Model | Accuracy |
|-------|----------|
| Original (Default) | 81.01% |
| Tuned (GridSearchCV) | 78.21% |
| Tuned (RandomizedSearchCV) | 80.45% |

### Key Takeaway
None of the tuned versions beat the original default (81.01%) on the test set. For a small, simple dataset like Titanic, the default Logistic Regression settings were already near-optimal, so tuning offered little room for improvement, a reminder that more tuning doesn't always mean better results.

Files

* `titanic_eda_weeks.ipynb` : EDA, cleaning, visualization, classification model, and model evaluation/tuning
* `house_price_regression.ipynb` : regression model for house price prediction

Tools Used

* Python, pandas, numpy, matplotlib, seaborn, scikit-learn
* Google Colab / Jupyter Notebook

Author
Zaka Saleem : [@zakasaleem35-jpg](https://github.com/zakasaleem35-jpg)
