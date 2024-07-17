# streamlit run app.py

import joblib
import pandas as pd
import streamlit as st

transf_pipl = joblib.load('./models/transf_pipl.sav')
mdl = joblib.load('./models/GBC_SMOTE.sav')

df = pd.read_csv('./data/train.csv')

def predict(data, transformer = transf_pipl, estimator = mdl):
    X_transf = transformer.transform(data)
    # Get the feature names
    feature_names = transformer.named_steps['Transformer'].get_feature_names_out()
    # Create dataframe with new feature names
    X_transf = pd.DataFrame(X_transf, columns= feature_names)

    # predict
    prediction = estimator.predict(X_transf)
    return prediction


#############
#############


# Streamlit application
st.header('Health Insurance Cross-Selling')
st.subheader('Prediction App')


# Inputs for each feature - enter your relevant features here:

# Gender
Gender= st.selectbox('Gender', ['Male', 'Female'])

# Age
Age = st.number_input('Age')

# Driving License
DrivingL = st.selectbox('Driving License', ['Yes', 'No'])
Driving_License = 1 if DrivingL == 'Yes' else 0

# Region Code
region_list = df['Region_Code'].unique().tolist()
Region_Code = st.selectbox('Region Code', region_list)

# Previously Insured
PrevInsu = st.selectbox('Previously Insured', ['Yes', 'No'])
Previously_Insured = 1 if PrevInsu == 'Yes' else 0

# Vehicle Age
Vehicle_Age = st.selectbox('Vehicle Age', ['< 1 Year','1-2 Year','> 2 Years'])

# Vehicle Damage
Vehicle_Damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])

# Annual Premium
Annual_Premium = st.number_input('Annual Premium')

# Policy_Sales_Channel
Policy_Sales_Channel_list = df['Policy_Sales_Channel'].unique().tolist()
Policy_Sales_Channel = st.selectbox('Policy Sales Channel', Policy_Sales_Channel_list)


##########
# Assemble inputs into DataFrame to match the input shape of trained model
input_data = pd.DataFrame([{'Gender': Gender,
                            'Age': Age,
                            'Driving_License': Driving_License,
                            'Region_Code': Region_Code,
                            'Previously_Insured': Previously_Insured,
                            'Vehicle_Age': Vehicle_Age,
                            'Vehicle_Damage': Vehicle_Damage,
                            'Annual_Premium': Annual_Premium,
                            'Policy_Sales_Channel': Policy_Sales_Channel
                            }])



if st.button('Predict'):
    prediction = predict(input_data)

    if prediction[0] == 0:
        st.write('Predicted Response: Customer will NOT ACCEPT the offer')
    else:
        st.write('Predicted Response: Customer will ACCEPT the offer')
