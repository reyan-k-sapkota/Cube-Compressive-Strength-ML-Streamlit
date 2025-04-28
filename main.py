import pickle as pkl
import pandas as pd

with open("Model_CS_GradientBoost.pkl", "rb") as f:
    model = pkl.load(f)

with open("Scaler_CS_GradientBoosting.pkl", "rb") as f:
    scaler = pkl.load(f)


example_features = {
    "Cement (component 1)(kg in a m^3 mixture)":None,
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": None,
    "Fly Ash (component 3)(kg in a m^3 mixture)":None,
    "Water  (component 4)(kg in a m^3 mixture)": None,
    "Superplasticizer (component 5)(kg in a m^3 mixture)":None,
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":None,
    "Fine Aggregate (component 7)(kg in a m^3 mixture)":None,
    "Age (day)":None,
}

keys = [ "Cement (component 1)(kg in a m^3 mixture)", "Blast Furnace Slag (component 2)(kg in a m^3 mixture)", "Fly Ash (component 3)(kg in a m^3 mixture)", "Water  (component 4)(kg in a m^3 mixture)",  "Superplasticizer (component 5)(kg in a m^3 mixture)", 
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)", "Fine Aggregate (component 7)(kg in a m^3 mixture)", "Age (day)"]

print ("Enter the value of Cement (component 1)(kg in a m^3 mixture)")
input1 = input ()
print ("Enter the value of Blast Furnace Slag (component 2)(kg in a m^3 mixture)")
input2 = input ()
print ("Enter the value of Fly Ash (component 3)(kg in a m^3 mixture)")
input3 = input ()
print ("Enter the value of Water (component 4)(kg in a m^3 mixture)")
input4 = input ()
print ("Enter the value of Superplasticizer (component 5)(kg in a m^3 mixture)")
input5 = input ()
print ("Enter the value of Coarse Aggregate (component 6)(kg in a m^3 mixture)")
input6 = input ()
print ("Enter the value of Fine Aggregate (component 7)(kg in a m^3 mixture)")
input7 = input ()
print ("Enter the value of Age (Curing age in days)")
input8 = input ()


values = [input1, input2, input3, input4, input5, input6, input7, input8]

for key, value in zip (keys, values):
    example_features[key] = value
print (example_features)

df_features = pd.DataFrame([example_features])
print (df_features)

#X_data = [[input1, input2, input3, input4, input5, input6, input7, input8]]
scaled_input = scaler.transform(df_features)
predicted_CS = model.predict (scaled_input)
print (predicted_CS)
