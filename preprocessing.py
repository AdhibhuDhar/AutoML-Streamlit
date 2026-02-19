import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(X,y):
    data=pd.concat([X,y],axis=1)
    data=data.replace("?",np.nan)
    data=data.fillna("missing")#we say it was nuking missing data while wiping the dataset
    #assuming last column as target
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    print("Class dist")
    print(y.value_counts())
    #filling missing values first
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

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)# now train and test has both 0 and 1 each aka retain same proportion of class labels
    return X_train,X_test,y_train,y_test           