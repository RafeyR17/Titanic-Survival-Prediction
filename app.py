import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config
st.set_page_config(page_title="Titanic Predictor", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Exploring the classic Titanic dataset with Machine Learning.")

# Load data and model
@st.cache_resource
def load_assets():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    model = joblib.load('titanic_model.pkl')
    
    # Hardcode the exact feature names from your training (order matters!)
    feature_names = [
        'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
        'FamilySize', 'IsAlone', 'Title',
        'emb_Q', 'emb_S'
    ]
    
    return train, test, model, feature_names

try:
    train_df, test_df, model, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictor"])

with tab1:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Data Head")
        st.dataframe(train_df.head())
    with col2:
        st.subheader("Data Statistics")
        st.write(train_df.describe())
    
    st.subheader("Survival Rate by Class & Sex")
    st.write(train_df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack())

with tab2:
    st.header("Survival Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax1, palette='viridis')
        ax1.set_title("Survival Rate by Sex")
        st.pyplot(fig1)
        
    with col4:
        fig2, ax2 = plt.subplots()
        sns.barplot(x='Pclass', y='Survived', data=train_df, ax=ax2, palette='magma')
        ax2.set_title("Survival Rate by Class")
        st.pyplot(fig2)

    st.subheader("Age Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(data=train_df, x="Age", hue="Survived", kde=True, element="step", ax=ax3)
    st.pyplot(fig3)

with tab3:
    st.header("Predict Survival")
    st.info("Enter passenger details below to see if they would have survived.")
    
    with st.expander("Passenger Information", expanded=True):
        col5, col6, col7 = st.columns(3)
        with col5:
            pclass = st.selectbox("Class (Pclass)", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
            sex = st.selectbox("Sex", ["male", "female"])
        with col6:
            age = st.slider("Age", 0, 80, 25)
            # SibSp and Parch for FamilySize calculation
            sibsp = st.number_input("Siblings/Spouses (SibSp)", 0, 10, 0)
        with col7:
            parch = st.number_input("Parents/Children (Parch)", 0, 10, 0)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

    if st.button("Predict Survival Status"):
        # 1. Feature Engineering (exact match to training)
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Sex → numeric (0=male, 1=female)
        sex_num = 1 if sex == "female" else 0
        
        # Embarked → dummies
        emb_Q = 1 if embarked == "Q" else 0
        emb_S = 1 if embarked == "S" else 0
        
        # Simple Title → numeric (match your training map)
        title_num = 0  # Mr default
        if sex == "female":
            title_num = 1 if age < 18 else 2   # Miss / Mrs
        elif age < 12:
            title_num = 3                      # Master
        
        # 2. Create input row with EXACT training column names & types
        input_row = {
            'pclass': pclass,
            'sex': sex_num,
            'age': age,
            'sibsp': sibsp,
            'parch': parch,
            'fare': 0.0,  # dummy/fallback - model needs it even if not used
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'Title': title_num,
            'emb_Q': emb_Q,
            'emb_S': emb_S
        }
        
        final_input = pd.DataFrame([input_row])
        
        # 3. Reorder columns to match training
        final_input = final_input[feature_names]
        
        # 4. Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]
        
        # Result
        st.divider()
        if prediction == 1:
            st.balloons()
            st.success(f"### 🎉 Survival Predicted!")
            st.markdown(f"Probability: **{probability:.1%}**")
        else:
            st.error(f"### 😔 Fatality Predicted")
            st.markdown(f"Survival probability: **{probability:.1%}**")
        
        st.caption("Disclaimer: This is a machine learning model based on historical data and not a definitive prediction.")
