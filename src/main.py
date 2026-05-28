import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/train.csv")

# Show first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

df["Married"] = df["Married"].fillna(df["Married"].mode()[0])

df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

df["Self_Employed"] = df["Self_Employed"].fillna(
    df["Self_Employed"].mode()[0]
)

df["LoanAmount"] = df["LoanAmount"].fillna(
    df["LoanAmount"].mean()
)

df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mean()
)

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mean()
)

# Convert categorical data
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

df["Married"] = df["Married"].map({
    "Yes": 1,
    "No": 0
})

df["Education"] = df["Education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["Self_Employed"] = df["Self_Employed"].map({
    "Yes": 1,
    "No": 0
})

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})

# Dependents column
df["Dependents"] = df["Dependents"].replace("3+", "3")

df["Dependents"] = pd.to_numeric(df["Dependents"])

# Property Area
df["Property_Area"] = df["Property_Area"].map({
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
})

# Remove Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# Split data
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Classification Report
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(6, 4))

sns.countplot(x="Loan_Status", data=df)

plt.title("Loan Approval Count")

plt.savefig("outputs/loan_approval_count.png")

plt.show()