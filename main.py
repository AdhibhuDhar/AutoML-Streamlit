import pandas as pd
from preprocessing import preprocess_data
from model_trainer import train_models
import os

#load dataset
#data=pd.read_csv("sample_data.csv")
print("AUTO_ML SYSTEM")
path=input("Enter dataset path : ")
ext=os.path.splitext(path)[1].lower()
if ext==".csv":
    data=pd.read_csv(path)
elif ext==".tsv":
    data=pd.read_csv(path,sep="\t")
elif ext==".txt":
    data=pd.read_csv(path,sep=None,engine="python")#instad of default C parser
#auto detect seperator
elif ext in [".xlsx",".xls"]:
    data=pd.read_excel(path)
else:
    print("Unsupported file")
    exit()
print("\nDataset Loaded Succesfully")
print(data.head())
target_col=input("\n Enter Target Column : ")
X=data.drop(target_col,axis=1)
y=data[target_col]

#pre-processing
X_train,X_test,y_train,y_test=preprocess_data(X,y)
train_models(X_train,X_test,y_train,y_test)
print("\nPreprocessing complete")
print("Training Shape",X_train.shape)
print("Testing shape:",X_test.shape)