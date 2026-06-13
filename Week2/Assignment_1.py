import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("Dataset 2.csv")    
                                # Part A :Dataset Understanding 
# Question 1:
print(df.head())

# Question 2:
rows = df.shape[0]
columns = df.shape[1]

print("Rows:", rows)
print("Columns:", columns)

# Question 3:
print(df.columns)

# Question 4:
print(df.dtypes)

                                   #Part B: Exploratory Data Analysis
# Question 6:                   
print( "\nAverage age of users:  " ,round(df["Age"].mean()))

# Question 7:
print( "\nAverage Watch hours per week : " ,round(df["WatchHoursPerWeek"].mean()))

# Question 8:
print( "Average monthly spending of users:  " ,round(df["MonthlySpend"].mean()))

# Question 9:
print(" Number of users in each subscription category : ", df["SubscriptionType"].value_counts())

# Question 10:
percentage = (df["SubscriptionRenewed"]== "Yes").mean()*100
print("\npercentage of users who renewed their subscriptions :" , percentage)

                                   # Part C: Data Preparation
# Question 11:
categorical_cols = ["Gender", "SubscriptionType", "FavoriteGenre", "SubscriptionRenewed"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

print(df.head())

# Question 12:
X = df.drop(columns=["UserID", "SubscriptionRenewed"])
y = df["SubscriptionRenewed"]

# Question 13:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

                                       # Part D: Decision Tree Classificatio
# Question 14:
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)

# Question 15:
print("\nModel Accuracy: ",accuracy_score(y_test,pred))

# Question 16:
print("\nConfusion matrix: ",confusion_matrix(y_test,pred))

                                           # Part E: K-Nearest Neighbors (KNN) 
# Question 17:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test,knn_pred))

# Question 18:
if accuracy_score(y_test,pred) > accuracy_score(y_test,knn_pred):
    print("Decision Tree performed better.")
elif accuracy_score(y_test,knn_pred)> accuracy_score(y_test,pred):
    print("KNN performed better.")
else:
    print("Both models performed equally well.")

                                              #  Part F: Linear Regression 

# Question 19:
i = df.drop(columns=["UserID", "MonthlySpend"])   # independent variables
j = df["MonthlySpend"]                            # dependent variable
i_train, i_test, j_train, j_test = train_test_split(
    i, j, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(i_train, j_train)
j_pred = lr_model.predict(i_test)

# Question 20 
new_user = pd.DataFrame({
    "Age": [25],
    "Gender": [1],
    "SubscriptionType": [2],
    "WatchHoursPerWeek": [15],
    "DevicesUsed": [2],
    "FavoriteGenre": [3],
    "AdClicks": [5],
    "SubscriptionRenewed": [0]   # add this column if used in training
})


# Predict monthly spending
predicted_spend = lr_model.predict(new_user)
print("Predicted Monthly Spend:", predicted_spend[0])







