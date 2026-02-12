
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# Create Simple Dataset
# ---------------------------
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 6, 1, 3, 7],
    "attendance": [50, 60, 65, 70, 80, 85, 90, 95, 55, 75, 88, 40, 68, 92],
    "previous_marks": [30, 40, 50, 60, 70, 75, 85, 90, 35, 65, 80, 20, 55, 88],
    "result": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# ---------------------------
# Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Test Accuracy
# ---------------------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# User Input
# ---------------------------
print("\nEnter Student Details:")
hours = float(input("Study Hours per day: "))
attendance = float(input("Attendance %: "))
marks = float(input("Previous Marks: "))
new_data = pd.DataFrame([[hours, attendance, marks]],
                        columns=["study_hours", "attendance", "previous_marks"])

prediction = model.predict(new_data)


if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
