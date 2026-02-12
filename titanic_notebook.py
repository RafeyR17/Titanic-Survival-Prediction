import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Setting random seed for reproducibility
RANDOM_STATE = 42

print("1. Loading Data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

print("\n2. Preprocessing & Feature Engineering...")
def preprocess_titanic_data(df):
    df = df.copy()
    # Handle Missing Values
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Create New Features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    
    # Drop Unnecessary Columns
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age', 'Fare']
    df = df.drop(drop_elements, axis=1)
    return df

train_processed = preprocess_titanic_data(train_df)
test_processed = preprocess_titanic_data(test_df)

print("\n3. Encoding Categorical Variables...")
train_encoded = pd.get_dummies(train_processed, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
test_encoded = pd.get_dummies(test_processed, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

X = train_encoded.drop('Survived', axis=1)
y = train_encoded['Survived']
X_test = test_encoded.reindex(columns=X.columns, fill_value=0)

print("\n4. Modeling & Hyperparameter Tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), 
                        param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X, y)

print(f"Best CV Score: {rf_grid.best_score_:.4f}")
final_model = rf_grid.best_estimator_

print("\n5. Generating Final Predictions...")
predictions = final_model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")

# Save the model and feature names for the app
joblib.dump(final_model, 'titanic_model.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')
print("Model and feature names saved for app.py")
print(submission.head())

# --- Visualizations ---
print("\n6. Saving Visualizations...")

# survival_analysis.png
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Sex', y='Survived', data=train_df, palette='viridis')
plt.title('Survival Rate by Sex')
plt.subplot(1, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=train_df, palette='magma')
plt.title('Survival Rate by Class')
plt.tight_layout()
plt.savefig('survival_analysis.png')
print("Survival analysis plot saved to 'survival_analysis.png'")

# feature_importance.png
plt.figure(figsize=(10, 6))
importances = pd.Series(final_model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved to 'feature_importance.png'")
