import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("placement_data.csv")

# Extract company list for dropdown
company_list = sorted(data["company"].unique().tolist())

# Features and target
X = data.drop("placed", axis=1)
y = data["placed"]

# Encode company
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
company_encoded = encoder.fit_transform(X[["company"]])
company_df = pd.DataFrame(company_encoded, columns=encoder.get_feature_names_out(["company"]))

X = X.drop("company", axis=1).reset_index(drop=True)
X = pd.concat([X, company_df], axis=1)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model & encoder
joblib.dump(model, "placement_model.pkl")
joblib.dump(encoder, "encoder.pkl")

def predict_new_student(data_dict):
    """Predict placement for new student data"""
    new_df = pd.DataFrame([data_dict])

    # Convert CGPA to percentage if needed
    if "tenth_type" in new_df.columns and new_df.at[0, "tenth_type"] == "cgpa":
        new_df.at[0, "tenth_score"] = new_df.at[0, "tenth_score"] * 9.5
    if "twelfth_type" in new_df.columns and new_df.at[0, "twelfth_type"] == "cgpa":
        new_df.at[0, "twelfth_score"] = new_df.at[0, "twelfth_score"] * 9.5

    # Drop type columns if they exist
    for col in ["tenth_type", "twelfth_type"]:
        if col in new_df.columns:
            new_df = new_df.drop(col, axis=1)

    # Encode company using saved encoder
    company_encoded = encoder.transform(new_df[["company"]])
    company_df = pd.DataFrame(company_encoded, columns=encoder.get_feature_names_out(["company"]))

    new_df = new_df.drop("company", axis=1).reset_index(drop=True)
    new_df = pd.concat([new_df, company_df], axis=1)

    # Ensure new_df has all columns the model expects
    for col in model.feature_names_in_:
        if col not in new_df.columns:
            new_df[col] = 0  # add missing columns with 0

    new_df = new_df[model.feature_names_in_]  # reorder columns

    # Predict
    prediction = model.predict(new_df)
    return "Will be Placed" if prediction[0] == 1 else "Will Not be Placed"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    video_file = None
    if request.method == "POST":
        data = {
            "tenth_score": float(request.form["tenth_score"]),
            "twelfth_score": float(request.form["twelfth_score"]),
            "cgpa": float(request.form["cgpa"]),
            "skills": int(request.form["skills"]),
            "internships": int(request.form["internships"]),
            "backlogs": int(request.form["backlogs"]),
            "company": request.form["company"]
        }
        prediction = predict_new_student(data)

        # Add video file based on prediction
        if prediction == "Will be Placed":
            video_file = "girl_yes.mp4"
        elif prediction == "Will Not be Placed":
            video_file = "girl_no.mp4"

    return render_template("index.html", prediction=prediction, companies=company_list, video_file=video_file)


if __name__ == "__main__":
    app.run(debug=True)
