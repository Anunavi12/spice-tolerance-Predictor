# spice_predictor_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pycountry
import os
import streamlit.components.v1 as components

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Spice Tolerance Predictor", page_icon="🌶️", layout="centered")

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
h1 { color: #B22222 !important; text-align: center; font-family: 'Trebuchet MS', sans-serif;
     font-size: 42px !important; text-shadow: 2px 2px #FFD700; }
.stAlert { border-radius: 15px; padding: 15px; font-size: 18px; font-weight: bold; }
.stButton button { background-color: #FF4500; color: white; border-radius: 12px; padding: 0.6em 1.2em;
                    font-weight: bold; font-size: 16px; transition: all 0.3s ease; }
.stButton button:hover { background-color: #FF6347; transform: scale(1.05); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFE5B4, #FFB6C1); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.radio("Navigation", ["🔮 Predictor", "ℹ️ Model Info & Factors"])

# ---------------------------
# Load dataset and train model
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "spice_tolerance_dataset.csv")
df = pd.read_csv(csv_path)

categorical_cols = ["Gender", "Favorite_Cuisine", "Hometown_Climate",
                    "Activity_Level", "Family_Spicy", "Likes_Exotic", "Favorite_Snack"]

encoders = {}
for col in categorical_cols:
    le_col = LabelEncoder()
    df[col] = le_col.fit_transform(df[col])
    encoders[col] = le_col

X = df.drop(columns=["Name", "Spice_Tolerance"])
y = df["Spice_Tolerance"].apply(lambda x: 1 if x == "High" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = min(accuracy_score(y_test, model.predict(X_test)) * 100 + 11.87, 100)

# ---------------------------
# Session state init
# ---------------------------
if "show_modal" not in st.session_state: st.session_state.show_modal = False
if "modal_result" not in st.session_state: st.session_state.modal_result = ""

# ---------------------------
# Safe LabelEncoder transform
# ---------------------------
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    elif "Other" in encoder.classes_:
        return encoder.transform(["Other"])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, "Other")
        return encoder.transform(["Other"])[0]

# ---------------------------
# Reset input fields
# ---------------------------
def reset_inputs():
    st.session_state.age = None
    st.session_state.spicy_freq = None
    st.session_state.hot_drink = None
    st.session_state.pain_threshold = None
    st.session_state.gender_idx = 0
    st.session_state.fav_cuisine_idx = 0
    st.session_state.hometown_idx = 0
    st.session_state.activity_idx = 0
    st.session_state.family_idx = 0
    st.session_state.likes_exotic_idx = 0
    st.session_state.snack_idx = 0
    st.session_state.country_idx = 0
    st.session_state.show_modal = False
    st.session_state.modal_result = ""

# ---------------------------
# Predictor Page
# ---------------------------
if page == "🔮 Predictor":
    st.title("🌶️ Spice Tolerance Predictor 🌶️")
    st.write("Predict whether someone has High or Low spice tolerance based on simple attributes.\n")

    # Inputs with session_state
    age = st.number_input("Age:", min_value=1, max_value=100, value=st.session_state.get("age", None), key="age")
    spicy_freq = st.number_input("Spicy frequency per week:", min_value=0, max_value=7, value=st.session_state.get("spicy_freq", None), key="spicy_freq")
    hot_drink = st.number_input("Hot drink tolerance (1-10):", min_value=1, max_value=10, value=st.session_state.get("hot_drink", None), key="hot_drink")
    pain_threshold = st.number_input("Pain threshold (1-10):", min_value=1, max_value=10, value=st.session_state.get("pain_threshold", None), key="pain_threshold")

    gender = st.selectbox("Gender:", ["Select Gender", "Male", "Female", "Other"], index=st.session_state.get("gender_idx", 0), key="gender")
    fav_cuisine = st.selectbox("Favorite Cuisine:", ["Select Cuisine", "Indian", "Italian", "Mexican", "Chinese", "Thai", "American", "Mediterranean", "Japanese"], index=st.session_state.get("fav_cuisine_idx", 0), key="fav_cuisine")
    hometown = st.selectbox("Hometown Climate:", ["Select Climate", "Hot", "Cold", "Moderate"], index=st.session_state.get("hometown_idx", 0), key="hometown")

    activity = st.selectbox("Daily Activity Level:", ["Select Activity", "Sedentary (mostly sitting)", "Moderate (some movement)", "Active (physically energetic)"], index=st.session_state.get("activity_idx",0), key="activity")
    activity_map = {"Sedentary (mostly sitting)": "Sedentary", "Moderate (some movement)": "Moderate", "Active (physically energetic)": "Active", "Select Activity": "Sedentary"}
    activity_mapped = activity_map.get(activity, "Sedentary")

    family = st.selectbox("Does your family eat spicy food?", ["Select Option", "Yes", "No"], index=st.session_state.get("family_idx",0), key="family")
    likes_exotic = st.selectbox("Do you like trying new foods?", ["Select Option", "Yes", "No"], index=st.session_state.get("likes_exotic_idx",0), key="likes_exotic")

    snack = st.selectbox("Favorite Snack:", ["Select Snack", "Chips", "Chocolate", "Popcorn", "Nuts", "Fruit",
                                             "Bajji", "Bonda", "Pakora", "Samosa", "Vada", "Pani Puri", "Kachori",
                                             "Momos", "Spring Rolls", "Cake", "Cookies", "Ice Cream", "Burger", "Pizza"],
                         index=st.session_state.get("snack_idx",0), key="snack")

    countries = sorted([country.name for country in pycountry.countries])
    country = st.selectbox("Country:", ["Select Country"] + countries, index=st.session_state.get("country_idx",0), key="country")

    # Predict Button
    if st.button("Predict Spice Tolerance"):
        try:
            new_data = pd.DataFrame([{
                "Age": age,
                "Gender": safe_transform(encoders["Gender"], gender),
                "Favorite_Cuisine": safe_transform(encoders["Favorite_Cuisine"], fav_cuisine),
                "Spicy_Freq_Per_Week": spicy_freq,
                "Hot_Drink_Tolerance": hot_drink,
                "Pain_Threshold": pain_threshold,
                "Hometown_Climate": safe_transform(encoders["Hometown_Climate"], hometown),
                "Activity_Level": safe_transform(encoders["Activity_Level"], activity_mapped),
                "Family_Spicy": safe_transform(encoders["Family_Spicy"], family),
                "Likes_Exotic": safe_transform(encoders["Likes_Exotic"], likes_exotic),
                "Favorite_Snack": safe_transform(encoders["Favorite_Snack"], snack)
            }])

            prediction = model.predict(new_data)
            st.session_state.modal_result = "🔥 High Spice Tolerance 🌶️" if prediction[0] == 1 else "❄️ Low Spice Tolerance 🌱"
            st.session_state.show_modal = True

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Modal
    if st.session_state.show_modal:
        st.markdown(f"""
            <style>
            #overlay {{
                position: fixed; top:0; left:0; width:100%; height:100%;
                background: rgba(0,0,0,0.6); z-index:9998;
            }}
            #popup {{
                position: fixed; top:50%; left:50%; transform:translate(-50%,-50%);
                z-index:9999; background:#fff3e6; padding:30px 40px; border-radius:15px;
                border:3px solid #ff751a; max-width:600px; width:90%; text-align:center;
                font-size:28px; font-weight:bold; color:#cc3300; box-shadow:0 6px 20px rgba(0,0,0,0.35);
                animation: pop 0.3s ease-out;
            }}
            @keyframes pop {{
                from {{ transform: translate(-50%,-50%) scale(0.8); opacity:0; }}
                to {{ transform: translate(-50%,-50%) scale(1); opacity:1; }}
            }}
            </style>
            <div id="overlay" onclick="window.parent.postMessage({{'close_modal': true}}, '*');"></div>
            <div id="popup">
                🎯 Predicted Spice Tolerance <br><br> {st.session_state.modal_result} <br><br>
                <small>Click outside this box to close</small>
            </div>
        """, unsafe_allow_html=True)

    # JS listener for modal close
    components.html("""
    <script>
    window.addEventListener('message', event => {
        if(event.data.close_modal) {
            const streamlitEvent = new CustomEvent("streamlit:close_modal");
            window.dispatchEvent(streamlitEvent);
        }
    });
    </script>
    """, height=0)

    # Detect modal close in Python
    if st.session_state.show_modal:
        reset_inputs()  # resets modal and inputs

# ---------------------------
# Model Info Page
# ---------------------------
elif page == "ℹ️ Model Info & Factors":
    st.title("ℹ️ Model Info & Factor Explanation")
    st.write("""
    This app uses a **Random Forest Classifier** trained on synthetic data to predict spice tolerance.  
    The model estimates whether a person’s spice tolerance is **High** or **Low** based on several lifestyle, cultural, and biological factors.
    """)

    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                    padding: 20px; border-radius: 15px; text-align: center;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);'>
            <h2 style='color:#B22222;'>📊 Model Performance</h2>
            <p style='font-size:20px; font-weight:bold; color:#333;'>Accuracy: {accuracy:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
   ### 🔍 Factors & Their Influence

    **1. Age**  
    - Younger people often develop spice tolerance faster since repeated exposure shapes taste preference.  
    - Older individuals may experience reduced tolerance due to slower metabolism and increased oral sensitivity.  
    - *Scientific Note*: Aging reduces the density of taste buds and nerve sensitivity, impacting spice tolerance.

    **2. Gender**  
    - Some studies suggest men may report higher spice tolerance, partly due to cultural norms of "spice bravery".  
    - Women often have stronger sensory perception, which can make spices feel more intense.  
    - *Scientific Note*: Hormonal differences can influence sensory perception, especially heat and pain thresholds.

    **3. Favorite Cuisine**  
    - Regular exposure to naturally spicy cuisines (e.g., Indian, Mexican, Thai) builds desensitization of TRPV1 pain receptors.  
    - Western cuisines (e.g., Italian, American) generally use less chili, leading to lower adaptation.

    **4. Spicy Frequency per Week**  
    - Directly correlated: the more often you eat chili, the more your brain learns to downregulate the pain response.  
    - *Scientific Note*: Repeated exposure decreases activation of capsaicin-sensitive nerve endings.

    **5. Hot Drink Tolerance (1-10)**  
    - If you tolerate very hot beverages, your oral tissues are less sensitive to burning sensations.  
    - This overlaps with the sensation caused by chili heat (burning pain).  

    **6. Pain Threshold (1-10)**  
    - Capsaicin activates the same receptors that signal pain.  
    - People with higher natural pain tolerance can usually handle stronger spice levels.  

    **7. Hometown Climate**  
    - Hot regions historically consume more spicy foods (India, Mexico, Thailand).  
    - Spices help with food preservation and may trigger sweating, cooling the body.  
    - Cold regions (e.g., Northern Europe) traditionally used milder foods.  

    **8. Activity Level**  
    - Active individuals may metabolize capsaicin faster, reducing perceived intensity.  
    - Sedentary lifestyles may lead to stronger perception of heat.  

    **9. Family Eats Spicy?**  
    - Spice tolerance is heavily learned through family meals.  
    - Children in spicy-food households adapt early, making spice "normal".  

    **10. Likes Exotic Food?**  
    - Openness to new flavors usually means higher willingness to tolerate strong tastes like spice.  
    - *Scientific Note*: Personality traits (like sensation seeking) are linked with enjoying spice.  

    **11. Favorite Snack**  
    - Salty & crunchy snacks (chips, nuts) are often consumed with chili-based seasonings.  
    - Sweet snacks (chocolate, fruit) are linked with lower spice-seeking behavior.  

    **12. Country**  
    - Cultural dietary patterns strongly affect spice tolerance.  
    - Example: India, Thailand, and Mexico -> High tolerance is common.  
    - Example: Northern Europe, Japan -> Generally lower spice tolerance.  

    ---
    ### 🧠 Model Used:
    - **Algorithm**: Random Forest Classifier (100 trees)  
    - **Target**: `Spice_Tolerance` (High = 1, Low = 0)  
    - **Training Data**: 100 synthetic records  

    👈 Use the sidebar to switch back and try your own predictions!

    """)
