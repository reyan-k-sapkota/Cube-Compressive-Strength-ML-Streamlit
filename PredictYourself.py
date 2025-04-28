import streamlit as st
import pickle as pkl
import pandas as pd

page_bg_color = """
<style>
    .stApp {
    background-color: #0093E9;
    background-image: linear-gradient(160deg, #0093E9 0%, #80D0C7 100%);

    background-attachment: fixed;
    color: black;
    }

    .stForm button {
        color: white;  /* Change button text color to white */
        background-color: #0078D4;  /* Optional: Set a custom background color */
        border-radius: 8px;  /* Optional: Change the button's border radius */
        padding: 10px 20px;  /* Optional: Add padding to the button */
    }

    .stForm button:hover {
        background-color: #005A8B;  /* Optional: Change the background color on hover */
        color: white;
        border:blue;    }

   

</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

st.title ("🔨PREDICT YOURSELF")
st.divider()

st.markdown(
    """
    **This functionality allows you to change the values of the dependent features and make predictions for the data of your choice.** The following features are allowed to be changed: 
    - Cement's mass (kg in a m^3 mixture)
    - Blast Furnace's Slag Quantity (kg in a m^3 mixture)
    - Fly Ash's Quantity (kg in a m^3 mixture)
    - Water's Weight (kg in a m^3 mixture)
    - Superplasticizer's Quantity (kg in a m^3 mixture)
    - Coarse Aggregate's Quantity (kg in a m^3 mixture)
    - Fine Aggregate's Quantity (kg in a m^3 mixture)
    - Days of curing
""")
st. divider()


with open("Model_CS_GradientBoost.pkl", "rb") as f:
    model1 = pkl.load(f)

with open("Scaler_CS_GradientBoosting.pkl", "rb") as f:
    scaler1 = pkl.load(f)

with open("Model_CS_RandomForest.pkl", "rb") as f:
    model2 = pkl.load(f)

with open("Scaler_CS_RandomForest.pkl", "rb") as f:
    scaler2 = pkl.load(f)

with open("Model_CS_CatBoost.pkl", "rb") as f:
    model3 = pkl.load(f)

with open("Scaler_CS_CatBoosting.pkl", "rb") as f:
    scaler3 = pkl.load(f)



algorithms = ["Gradient Boost Regressor", "CatBoost Regressor", "RandomForest Regressor"]



with st.form("prediction_form"):
    
    cement = st.slider("Cement's Weight. (kg in a m^3 mixture)", min_value=1, max_value=700, value=1,)

    slag = st.slider("Blast Furnace Slag's Weight. (kg in a m^3 mixture)", min_value=0, max_value=150, value=1, )
    
    fly_ash = st.slider("Fly Ash's Weight. (kg in a m^3 mixture)", min_value=0, max_value=150, value=1,)
    
    water = st.slider("Water's Weight. (kg in a m^3 mixture)", min_value=0, max_value=200, value=1)

    super_plasticizer = st.slider("SuperPlasticizer's Weight. (kg in a m^3 mixture)", min_value=0, max_value=75, step=1)

    coarse_aggregate = st.slider("Coarse Aggregate Weight. (kg in a m^3 mixture)", min_value=750, max_value=1500, value=1 )
    
    fine_aggregate = st.slider("Fine Aggregate's Weight. (kg in a m^3 mixture)", min_value=300, max_value=1200, value=1)

    age = st.slider("Concrete's curing age (in days)", min_value=3, max_value=360, value=1)


    c1, c2, c3 = st.columns(3)
    with c1:
        submit_button1 = st.form_submit_button(label="Predict using Gradient Boost Regressor")
    
    with c2:
        submit_button2 = st.form_submit_button(label="Predict using CatBoost Regressor")

    with c3:
        submit_button3 = st.form_submit_button(label="Predict using Random Forest Regressor")
    


example_features = {
    "Cement (component 1)(kg in a m^3 mixture)":cement,
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": slag,
    "Fly Ash (component 3)(kg in a m^3 mixture)":fly_ash,
    "Water  (component 4)(kg in a m^3 mixture)": water,
    "Superplasticizer (component 5)(kg in a m^3 mixture)":super_plasticizer,
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":coarse_aggregate,
    "Fine Aggregate (component 7)(kg in a m^3 mixture)":fine_aggregate,
    "Age (day)":age,
}

input_dataframe = pd.DataFrame([example_features])


if submit_button1:
    scaled_input = scaler1.transform(input_dataframe)
    predicted_CS = model1.predict (scaled_input)
    st.write(f"Your chosen algorithm is **{algorithms[0]}**")
    st.write(f"Predicted Compressive Strength is **{predicted_CS}** MPa")
    st.write (f"The R2 Value for this **{algorithms[0]}** model is 0.9084, MAE is 3.022, and RMSE is 4.85")

if submit_button2:
    scaled_input = scaler2.transform(input_dataframe)
    predicted_CS = model2.predict (scaled_input)
    st.write(f"Your chosen algorithm is **{algorithms[1]}**")
    st.write(f"Predicted Compressive Strength is **{predicted_CS}** MPa")
    st.write (f"The R2 Value for this **{algorithms[1]}** model is 0.923, MAE is 2.9024, and RMSE is 4.432.")

    
if submit_button3:
    scaled_input = scaler3.transform(input_dataframe)
    predicted_CS = model3.predict (scaled_input)
    st.write(f"Your chosen algorithm is **{algorithms[2]}**")
    st.write(f"Predicted Compressive Strength is **{predicted_CS}** MPa")
    st.write (f"The R2 Value for this **{algorithms[2]}** model is 0.854, MAE is 4.757, and RMSE is 6.1183")


st.divider()

github_icon_url = "https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg"
footer = """
    <style>
        .footer {
            text-align: center;
            font-size: 15px;
            color: #fff;
            margin-top: 50px;
            background-color: #1E90FF;  /* Blue background */
            padding: 20px;
            border-radius: 8px;
        }
        .footer a {
            color: #fff;
            text-decoration: none;
            font-weight: bold;
        }
        .footer i {
            font-size: 18px;
            margin-right: 8px;
        }
        .footer p {
            margin: 10px 0;
        }
    </style>
    <div class="footer">
        <p>Made by <br>Reyan Kumar Sapkota</br></p>
        <p><i>Engineers Without Borders, Pulchowk Chapter</i></p>
        <p>
            <a href="https://github.com/reyan-k-sapkota" target="_blank">
                <i class="fab fa-github"></i> GitHub
            </a>
        </p>
    </div>
"""

# Display the footer in Streamlit
st.markdown(footer, unsafe_allow_html=True)