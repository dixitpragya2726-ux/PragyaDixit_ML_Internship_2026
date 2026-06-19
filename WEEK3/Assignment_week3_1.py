
# PART A: UNDERSTANDING THE DATASET
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load Dataset

df = pd.read_csv("Week1/WEEK2/Week3/agriculture_yield_dataset.csv")
print("="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

# Q1. DATASET OVERVIEW
print("\nQ1. DATASET OVERVIEW")
# Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# Column Names
print("\nColumn Names:")
print(df.columns.tolist())

# First 10 Records
print("\nFirst 10 Records:")
print(df.head(10))

# Q2. DATA TYPES AND MISSING VALUES

print("\nQ2. DATA TYPES AND MISSING VALUES")

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Columns with Missing Values
missing_cols = df.columns[df.isnull().sum() > 0]

if len(missing_cols) > 0:
    print("\nColumns with Missing Values:")
    print(missing_cols.tolist())
else:
    print("\nNo Missing Values Found")

# Q3. DESCRIPTIVE STATISTICS

print("\nQ3. DESCRIPTIVE STATISTICS")
summary = df.describe()
print("\nSummary Statistics:")
print(summary)

# Highest Mean
highest_mean_feature = summary.loc["mean"].idxmax()

print("\nFeature with Highest Mean:")
print(highest_mean_feature)

# Highest Standard Deviation
highest_std_feature = summary.loc["std"].idxmax()

print("\nFeature with Highest Standard Deviation:")
print(highest_std_feature)

# PART B: EXPLORATORY DATA ANALYSIS (EDA)
# Q4. DISTRIBUTION ANALYSIS

print("\nQ4. DISTRIBUTION ANALYSIS")

plt.figure(figsize=(6,4))
plt.hist(df["rainfall_mm"], bins=20)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["temperature_c"], bins=20)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["fertilizer_kg"], bins=20)
plt.title("Fertilizer Distribution")
plt.xlabel("Fertilizer")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()


# Q5. CROP TYPE ANALYSIS

print("\nQ5. CROP TYPE ANALYSIS")

crop_count = df["crop_type"].value_counts()

print("\nCrop Frequency:")
print(crop_count)

plt.figure(figsize=(8,5))
sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.xticks(rotation=45)
plt.show()

print("\nMost Frequent Crop Type:")
print(df["crop_type"].mode()[0])

# Q6. SOIL TYPE ANALYSIS

print("\nQ6. SOIL TYPE ANALYSIS")

soil_count = df["soil_type"].value_counts()

print("\nSoil Frequency:")
print(soil_count)

plt.figure(figsize=(8,5))
sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.xticks(rotation=45)
plt.show()

print("\nMost Common Soil Type:")
print(df["soil_type"].mode()[0])


# Q7. YIELD DISTRIBUTION

print("\nQ7. YIELD DISTRIBUTION")

plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# Outlier Detection
Q1 = df["yield_ton_per_hectare"].quantile(0.25)
Q3 = df["yield_ton_per_hectare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["yield_ton_per_hectare"] < lower) |
    (df["yield_ton_per_hectare"] > upper)
]

print("\nNumber of Outliers:")
print(len(outliers))

# Q8. SCATTER PLOT ANALYSIS

print("\nQ8. SCATTER PLOT ANALYSIS")

plt.figure(figsize=(6,4))
plt.scatter(df["rainfall_mm"],
            df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["fertilizer_kg"],
            df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

corr_rainfall = df["rainfall_mm"].corr(
    df["yield_ton_per_hectare"]
)

corr_fertilizer = df["fertilizer_kg"].corr(
    df["yield_ton_per_hectare"]
)

print("\nCorrelation with Yield")

print("Rainfall:", corr_rainfall)
print("Fertilizer:", corr_fertilizer)

if abs(corr_rainfall) > abs(corr_fertilizer):
    print("\nRainfall has stronger relationship with Yield")
else:
    print("\nFertilizer has stronger relationship with Yield")


# Q9. CORRELATION ANALYSIS


print("\nQ9. CORRELATION ANALYSIS")

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

yield_corr = corr_matrix[
    "yield_ton_per_hectare"
].sort_values(ascending=False)

print("\nTop Features Correlated With Yield:")
print(yield_corr)


# Q10. GROUP BASED ANALYSIS

print("\nQ10. GROUP BASED ANALYSIS")

crop_avg = df.groupby("crop_type")[
    "yield_ton_per_hectare"
].mean()

print("\nAverage Yield by Crop Type:")
print(crop_avg)

soil_avg = df.groupby("soil_type")[
    "yield_ton_per_hectare"
].mean()

print("\nAverage Yield by Soil Type:")
print(soil_avg)

print("\nHighest Yield Crop Type:")
print(crop_avg.idxmax())

print("\nHighest Yield Soil Type:")
print(soil_avg.idxmax())



# ---------------------------------------PART C: DATA PREPARATION----------------------------------------------------

# Q11. FEATURE ENCODING

print("\nQ11. FEATURE ENCODING")

categorical_cols = df.select_dtypes(
    include="object"
).columns

print("\nCategorical Columns:")
print(categorical_cols.tolist())

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nFirst Five Rows of Encoded Dataset:")
print(df_encoded.head())


# Q12. FEATURE SELECTION

print("\nQ12. FEATURE SELECTION")

target = "yield_ton_per_hectare"

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

print("\nTarget Variable:")
print(target)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# PART D: MACHINE LEARNING

# Q13. TRAIN TEST SPLIT

print("\nQ13. TRAIN TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nShapes")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Q14. LINEAR REGRESSION MODEL

print("\nQ14. LINEAR REGRESSION MODEL")

model = LinearRegression()

model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

highest_positive = coefficients.loc[
    coefficients["Coefficient"].idxmax()
]

print("\nFeature with Highest Positive Coefficient:")
print(highest_positive)
