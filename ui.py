import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from model_trainer import train_models

st.set_page_config(page_title="AutoML Pipeline",layout="centered")
st.title("AutoML Model Trainer")
st.write("Upload Dataset->Select Target->Train Models")

#upload
file=st.file_uploader("Upload CSV/TSV/Excel file",type=["csv","tsv","xlsx"])
if file is not None:
    if file.name.endswith(".csv"):
        data=pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        data=pd.read_csv(file,sep="\t")
    elif file.name.endswith(".xlsx"):
        data=pd.read_excel(file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    #target
    target_col=st.selectbox("Select Target Column",data.columns)
    if st.button("🔮Train Models"):
        X=data.drop(target_col,axis=1)
        y=data[target_col]
        X_train,X_test,y_train,y_test=preprocess_data(X,y)
        best_model,best_score=train_models(X_train,X_test,y_train,y_test)
        st.success(f"Best Model : {best_model}")
        st.info(f"Accuracy : {round(best_score,3)}")
