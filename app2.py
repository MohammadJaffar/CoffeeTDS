import streamlit as st
from datetime import timedelta
import pandas as pd
import pickle
import sklearn

# Load the trained model
def new_func():
    with open(r'CoffeeTDS\Model\lineareg_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = new_func()

# Streamlit page design
st.set_page_config(page_title="Coffee TDS Calculator ☕", page_icon="☕", layout="wide")

# Add custom CSS for the entire page
st.markdown(
    """ 
    <style>
        /* Page background */
        body {
            background-color: #FDF3E6; /* Light coffee beige */
            color: #4B3832; /* Coffee brown font color */
            font-family: 'Verdana', sans-serif;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #F4EDE6; /* Sidebar light coffee beige */
        }

        /* Input fields */
        .stSidebar input, .stSidebar .st-slider {
            color: #4B3832; /* Text color for inputs */
        }

        /* Buttons */
        button {
            background-color: #D3B8AE !important; /* Soft coffee tan */
            color: #4B3832 !important; /* Coffee brown text */
            font-family: 'Verdana', sans-serif;
            border-radius: 8px !important; /* Rounded corners for buttons */
            padding: 8px !important;
            font-size: 16px !important;
        }

        button:hover {
            background-color: #C6A49A !important; /* Slightly darker hover effect */
        }

        /* TDS box styling */
        .tds-box {
            background-color: #FFF8E7; /* Light beige */
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }

        .tds-box h2 {
            color: #6B4226; /* Coffee brown */
            font-family: 'Georgia', serif; /* Elegant font for TDS display */
        }

        /* Extracted strength messages */
        .under-extracted {
            color: #A0522D; /* Weak coffee brown */
        }
        .balanced {
            color: #228B22; /* Balanced green */
        }
        .over-extracted {
            color: #B22222; /* Strong red */
        }

        /* Page title */
        h1, h3, p {
            text-align: center;
            font-family: 'Georgia', serif; /* Elegant font for titles */
            color: #4B3832; /* Coffee brown */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown(
    """
    <h1>Coffee TDS Calculator ☕</h1>
    <p>Welcome to the Coffee TDS Calculator! Enter the details of your coffee brewing process and let's make better coffee.</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar Input Fields
st.sidebar.header("Enter your coffee brewing details")

def seconder(brew_time):
    mins, secs = map(float, brew_time.split(':'))
    td = timedelta(minutes=mins, seconds=secs)
    return td.total_seconds()

# User inputs for the coffee brewing process
coffee_dose = st.sidebar.number_input("Coffee Dose (grams)", min_value=1, value=15, step=1)
water_temp = st.sidebar.slider("Water Temperature (°C)", min_value=60, max_value=100, value=92)
water_vol = st.sidebar.number_input("Water Volume (ml)", min_value=50, value=225, step=10)
num_pours = st.sidebar.number_input("Number of Pours", min_value=1, value=3, step=1)
brew_time = st.sidebar.text_input("Brew Time (min:sec)", value="2:00")

# Button to trigger prediction
if st.button("Calculate TDS"):
    try:
        brew_time_seconds = seconder(brew_time)

        # Prepare input data
        data = {
            "dose (g)": [coffee_dose],
            "water temp ( C )": [water_temp],
            "water vol (ml)": [water_vol],
            "number of pours": [num_pours],
            "seconds": [brew_time_seconds],
        }
        input_df = pd.DataFrame(data)

        # Ensure all features are present
        for feature in model.feature_names_in_:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default value

        # Reorder columns to match the model's training order
        input_df = input_df[model.feature_names_in_]

        # Predict coffee strength
        input_predict = model.predict(input_df)

        # Display Results
        tds = float(input_predict[0])
        st.markdown(
            f"""
            <div class="tds-box">
                <h2>TDS: {tds:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if tds < 1.25:
            st.markdown(
                """
                <h3 class="under-extracted">UnderExtracted (Weak)</h3>
                """,
                unsafe_allow_html=True,
            )
        elif 1.25 <= tds <= 1.45:
            st.markdown(
                """
                <h3 class="balanced">Well Extracted (Balanced)</h3>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <h3 class="over-extracted">OverExtracted (Strong)</h3>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
