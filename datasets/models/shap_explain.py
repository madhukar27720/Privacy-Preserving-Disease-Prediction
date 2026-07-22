import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/heart_model.pkl")

# Load dataset
df = pd.read_csv("datasets/heart.csv")

X = df.drop("target", axis=1)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary Plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary.png")

print("SHAP explanation generated successfully!")
