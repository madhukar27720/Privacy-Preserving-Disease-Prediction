import joblib

model = joblib.load("models/heart_model.pkl")

sample = [[
    52,
    1,
    0,
    125,
    212,
    0,
    1,
    168,
    0,
    1.0,
    2,
    2,
    3
]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
