import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# 2. Basic EDA (Internal check)
print("\nMissing values in Train:")
print(train_df.isnull().sum())

# 3. Feature Engineering & Preprocessing Function
def preprocess_data(df):
    # Fill missing values
    # Age: Fill with median grouped by Sex and Pclass
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Embarked: Fill with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fare: Fill with median (needed for test set)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # New features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Extract Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    df = df.drop(drop_cols, axis=1)
    
    return df

# Apply preprocessing
train_processed = preprocess_data(train_df.copy())
test_processed = preprocess_data(test_df.copy())

# Label Encoding for Title and Sex
# One-hot encoding as requested for final code, but for quick check we can use get_dummies
train_processed = pd.get_dummies(train_processed, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
test_processed = pd.get_dummies(test_processed, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# Ensure same columns in test as in train (except Survived)
test_processed = test_processed.reindex(columns=train_processed.columns.drop('Survived'), fill_value=0)

# 4. Prepare Splits
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeling comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\nModel Comparison (CV Score):")
for name, model in models.items():
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    print(f"{name}: {cv_score:.4f}")

# 6. Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"\nBest Params: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# 7. Final Prediction
best_model.fit(X, y)
predictions = best_model.predict(test_processed)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv('submission_test.csv', index=False)
print("\nSubmission file 'submission_test.csv' created.")
print(submission.head())
