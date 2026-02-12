# Titanic Survival Prediction

Interactive web app built with Streamlit that predicts whether a passenger would survive the Titanic disaster.

## Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Streamlit (UI)
- Random Forest Classifier
- Joblib (model saving)

## Features
- Survival rate visualizations (by sex, class, age)
- Interactive predictor with probability output

## Live Demo
(add later when deployed)

## How to run locally
```bash
git clone https://github.com/RafeyR17/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
