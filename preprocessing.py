import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(X,y):
    X=X.replace("?",np.nan)
    for col in X.columns:
        if X[col].dtype=="object":
            X[col]=X[col].astype(str)
            X[col]=X[col].fillna("Unknown")
        else:
            X[col]=X[col].fillna(X[col].mean())

    #encoding in categotrical columns
    for col in X.columns:
        if X[col].dtype=="object":
            le=LabelEncoder()#convert catg data to 0-n-1 class
            X[col]=le.fit_transform(X[col])

    #encode target if needed
    if y.dtype=="object":
        le=LabelEncoder()
        y=le.fit_transform(y)
        print("Class Distribution")
        print(pd.Series(y).value_counts())

    
    test_size = 0.2
    stratify_arg = None
    if len(pd.Series(y).unique()) < 20:
        counts = pd.Series(y).value_counts()
        # require at least two samples per class
        if counts.min() >= 2:
            stratify_arg = y
        else:
            # if some class is too rare, skip stratification
            stratify_arg = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_arg,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    return X_train,X_test,y_train,y_test           