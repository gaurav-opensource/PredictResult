
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create a synthetic dataset
np.random.seed(42)
n_students = 1000
data = {
    'exam_score': np.random.normal(60, 15, n_students),
    'attendance': np.random.uniform(50, 100, n_students),
    'study_hours': np.random.uniform(0, 10, n_students),
    'extracurricular': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
    'pass_fail': np.where(
        (np.random.normal(60, 15, n_students) > 50) & 
        (np.random.uniform(50, 100, n_students) > 60) & 
        (np.random.uniform(0, 10, n_students) > 2), 1, 0
    )
}
df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Visualize pass/fail distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='pass_fail', data=df)
plt.title('Pass/Fail Distribution')
plt.savefig('pass_fail_distribution.png')
plt.close()

# Step 3: Data Preprocessing
# Define features and target
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Step 6: Interactive User Input for Prediction
def predict_pass_fail(exam_score, attendance, study_hours, extracurricular):
    input_data = np.array([[exam_score, attendance, study_hours, extracurricular]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return "Pass" if prediction[0] == 1 else "Fail"

# Interactive input loop
while True:
    print("\nEnter student details for pass/fail prediction (or type 'exit' to quit):")
    try:
        exam_score_input = input("Exam Score (0-100): ")
        if exam_score_input.lower() == 'exit':
            break
        exam_score = float(exam_score_input)
        if not 0 <= exam_score <= 100:
            print("Error: Exam score must be between 0 and 100.")
            continue

        attendance_input = input("Attendance Percentage (0-100): ")
        attendance = float(attendance_input)
        if not 0 <= attendance <= 100:
            print("Error: Attendance must be between 0 and 100.")
            continue

        study_hours_input = input("Study Hours per Day (0-24): ")
        study_hours = float(study_hours_input)
        if not 0 <= study_hours <= 24:
            print("Error: Study hours must be between 0 and 24.")
            continue

        extracurricular_input = input("Participates in Extracurricular Activities? (0 for No, 1 for Yes): ")
        extracurricular = int(extracurricular_input)
        if extracurricular not in [0, 1]:
            print("Error: Extracurricular must be 0 or 1.")
            continue

        # Make prediction
        result = predict_pass_fail(exam_score, attendance, study_hours, extracurricular)
        print(f"\nPrediction for student (Exam: {exam_score}, Attendance: {attendance}%, "
              f"Study Hours: {study_hours}, Extracurricular: {extracurricular}): {result}")

    except ValueError:
        print("Error: Please enter valid numerical values.")

# Save the model (optional)
import joblib
joblib.dump(model, 'pass_fail_model.pkl')
joblib.dump(scaler, 'scaler.pkl')