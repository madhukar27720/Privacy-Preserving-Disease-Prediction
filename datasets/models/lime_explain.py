import pandas as pd
import joblib

from lime.lime_tabular import LimeTabularExplainer

# Load model
model = joblib.load("models/heart_model.pkl")

# Load dataset
df = pd.read_csv("datasets/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Create Explainer
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    mode="classification"
)

# Explain first sample
exp = explainer.explain_instance(
    X.iloc[0].values,
    model.predict_proba,
    num_features=6
)

exp.save_to_file("lime_explanation.html")

print("LIME explanation generated successfully!")
