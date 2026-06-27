import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================
# Question 1
# ==========================

df = pd.read_csv("Week1/WEEK2/WEEK4/Loan prediction.csv")

print("First 10 Records")
print(df.head(10))

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

print("\nTarget Variable")
print(df["Loan_Status"].head())

# ==========================
# Question 2
# ==========================

df.drop("Loan_ID", axis=1, inplace=True)

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

encoder = LabelEncoder()

categorical = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nData Preprocessing Completed")

# ==========================
# Question 3 : Exploratory Data Analysis
# ==========================

plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Status Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Gender", hue="Loan_Status", data=df)
plt.title("Gender vs Loan Status")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Education", hue="Loan_Status", data=df)
plt.title("Education vs Loan Status")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["ApplicantIncome"], bins=30, kde=True)
plt.title("Applicant Income Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ==========================
# Question 4 : Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# Question 5 : Model Building
# ==========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, accuracy, precision, recall, f1])

results = pd.DataFrame(
    results,
    columns=["Model","Accuracy","Precision","Recall","F1 Score"]
)

print(results)

# ==========================
# Question 6 : Stratified 5-Fold Cross Validation
# ==========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n", name)
    print("Cross Validation Scores :", scores)
    print("Mean Accuracy :", scores.mean())
    print("Standard Deviation :", scores.std())


# ==========================
# Question 7 : Hyperparameter Tuning
# ==========================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("Tuned Model Accuracy :", accuracy_score(y_test, y_pred))


# ==========================
# Question 8 : Bias Variance Tradeoff
# ==========================

depths = [2, 5, 15]

train_accuracy = []
test_accuracy = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

    print("\nDepth :", depth)
    print("Training Accuracy :", train_acc)
    print("Testing Accuracy :", test_acc)

plt.figure(figsize=(6,4))
plt.plot(depths, train_accuracy, marker='o', label="Training Accuracy")
plt.plot(depths, test_accuracy, marker='o', label="Testing Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()