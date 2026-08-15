import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_performance.csv")

# Features and target
X = data[["study_hours", "attendance", "previous_score"]]
y = data["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Example prediction
student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [90],
    "previous_score": [75]
})

predicted_score = model.predict(student)

print(f"Predicted Final Score: {predicted_score[0]:.2f}")
