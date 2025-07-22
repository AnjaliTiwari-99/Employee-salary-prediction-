import streamlit as st
import pickle
import numpy as np

# Load model, encoders, and scaler
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Title
st.title("Income Category Prediction")
st.subheader("Enter Individual's Information:")

# Category Mappings for UI dropdowns
category_mappings = {
    "gender": ['Male', 'Female'],
    "workclass": ['Private', 'Local-gov', '?', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
    "marital-status": ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse'],
    "occupation": ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', '?', 'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical',
                   'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'],
    "relationship": ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'],
    "race": ['Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'],
    "native-country": ['United-States', '?', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Ireland', 'Germany', 'Philippines',
                       'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia',
                       'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua', 'Honduras',
                       'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago',
                       'Outlying-US(Guam-USVI-etc)', 'France', 'Holand-Netherlands'],
    "education": ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school', '7th-8th', 'Bachelors', 'Masters', 'Doctorate',
                  '5th-6th', 'Assoc-voc', '9th', '12th', '1st-4th', 'Preschool']
}

# Collect input features from user
user_inputs = {}

# Numeric fields
user_inputs["age"] = st.number_input("Age", min_value=0, max_value=100, value=30)
user_inputs["fnlwgt"] = st.number_input("fnlwgt", min_value=0, value=100000)
user_inputs["educational-num"] = st.number_input("Educational Number", min_value=0, max_value=20, value=10)
user_inputs["capital-gain"] = st.number_input("Capital Gain", min_value=0, value=0)
user_inputs["capital-loss"] = st.number_input("Capital Loss", min_value=0, value=0)
user_inputs["hours-per-week"] = st.number_input("Hours per Week", min_value=0, max_value=100, value=40)

# Encoded categorical fields
for feature, labels in category_mappings.items():
    label_to_index = {label: i for i, label in enumerate(labels)}
    selected_label = st.selectbox(f"{feature.replace('-', ' ').title()}", labels)
    user_inputs[feature] = label_to_index[selected_label]

# Predict button
if st.button("Predict Income Category"):
    # Arrange features in order used during training
    input_order = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'gender', 'capital-gain',
                   'capital-loss', 'hours-per-week', 'native-country']

    input_array = np.array([user_inputs[feature] for feature in input_order]).reshape(1, -1)

    # Scale numerical features
    input_array_scaled = scaler.transform(input_array)

    # Predict probability and class
    y_pred_proba = model.predict_proba(input_array_scaled)[0][1]
    y_pred = int(y_pred_proba > 0.5)

    # Output
    label = ">50K" if y_pred == 1 else "<=50K"
    st.markdown(f"### Predicted Income Category: **{label}**")


# Optional: Show category mappings in sidebar
st.sidebar.title("Category Index Mappings")
for feature, labels in category_mappings.items():
    mapping_text = "\n".join([f"{i}: {label}" for i, label in enumerate(labels)])
    st.sidebar.markdown(f"**{feature.title()}**\n```\n{mapping_text}\n```")

