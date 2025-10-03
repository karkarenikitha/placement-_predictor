# placement_model.py

# -------------------------------
# 1. Import libraries
# -------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# 2. Load dataset
# -------------------------------
df = pd.read_csv("placement_sample_500.csv")

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Dataset info ===")
print(df.info())

print("\n=== Summary statistics ===")
print(df.describe())

print("\n=== Placement counts ===")
print(df['placed'].value_counts())

# -------------------------------
# 3. Preprocessing
# -------------------------------
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")  # added handle_unknown

# One-hot encode 'company'
company_encoded = encoder.fit_transform(df[['company']])
company_df = pd.DataFrame(company_encoded, columns=encoder.get_feature_names_out(['company']))

df = df.drop('company', axis=1)
df = pd.concat([df, company_df], axis=1)

# Features and labels
X = df.drop(['s_id', 'placed'], axis=1)
y = df['placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Preprocessing complete!")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

# -------------------------------
# 4. Train models
# -------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    if name == "Random Forest":  # Save Random Forest as best
        best_model = model

# -------------------------------
# 5. Save the best model & encoder
# -------------------------------
joblib.dump(best_model, "placement_model.pkl")
print("\n✅ Random Forest model saved as 'placement_model.pkl'")

joblib.dump(encoder, "encoder.pkl")
print("✅ Encoder saved as 'encoder.pkl'")

# -------------------------------
# 6. Function to predict new student
# -------------------------------
def predict_new_student(data_dict):
    """
    Predict placement for a new student.
    data_dict keys:
      'tenth_score', 'tenth_type', 'twelfth_score', 'twelfth_type',
      'cgpa', 'skills', 'internships', 'backlogs', 'company'
    """
    # Load saved model and encoder
    model = joblib.load("placement_model.pkl")
    encoder = joblib.load("encoder.pkl")

    # Convert input dictionary to DataFrame
    new_df = pd.DataFrame([data_dict])

    # Convert 10th and 12th scores to percentage if needed
    if new_df.at[0, "tenth_type"] == "cgpa":
        new_df.at[0, "tenth_score"] = new_df.at[0, "tenth_score"] * 9.5
    if new_df.at[0, "twelfth_type"] == "cgpa":
        new_df.at[0, "twelfth_score"] = new_df.at[0, "twelfth_score"] * 9.5

    # Drop type columns (not needed for model)
    new_df = new_df.drop(["tenth_type", "twelfth_type"], axis=1)

    # Encode company
    company_encoded = encoder.transform(new_df[['company']])
    company_df = pd.DataFrame(company_encoded, columns=encoder.get_feature_names_out(['company']))

    new_df = new_df.drop('company', axis=1)
    new_df = pd.concat([new_df, company_df], axis=1)

    # Predict
    prediction = model.predict(new_df)
    return "Placed" if prediction[0] == 1 else "Not Placed"



