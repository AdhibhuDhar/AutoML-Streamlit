from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,confusion_matrix,ConfusionMatrixDisplay
def train_models(X_train,X_test,y_train,y_test):
    if y_train.dtype=="float" or len(np.unique(y_train))>20:
        problem_type="regression"
    else:
        problem_type="classification"
    if problem_type=="classification":
         models={
        "Logistic Regression":LogisticRegression(max_iter=500), #model dict
        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(),
        "SVM":SVC(),
        "KNN":KNeighborsClassifier(),
        "Naive Bayes":GaussianNB()
    }   
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        models={
            "Linear Regression":LinearRegression(),
            "Decision Tree":DecisionTreeRegressor(),
            "Random Forest":RandomForestRegressor(),
            "SVR":SVR(),
            "KNN Regressor":KNeighborsRegressor()
        }
   
    best_model=None
    best_score=0
    print("\nTraining Models..\n")
    model_names=[]
    model_scores=[]
    for name,model in models.items():
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        #score=accuracy_score(y_test,preds)
        if problem_type=="classification":
            score=accuracy_score(y_test,preds)
        else:
            score=r2_score(y_test,preds)
        print(f"{name} score:{round(score,3)}")
        model_names.append(name)
        model_scores.append(score) 
    if score>best_score:
        best_score=score
        best_model=model
    print("Model saved as best_model.pkl")  
    print("\nBest model:",best_model,"Score:",round(best_score,3))
    print("\n===MODEL PERFORMANCE===")
    for i in range(len(model_names)):
        print(f"{model_names[i]} : {round(model_scores[i],3)}")
        
    print("\nBest Model:",best_model,"Score:",round(best_score,3)) 
    joblib.dump(best_model,"best_model.pkl")   

    plt.figure(figsize=(6,4))
    plt.bar(model_names,model_scores)
    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.show() 
    if problem_type=="classification":
        cm=confusion_matrix(y_test,best_model.predict(X_test))
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.show() 
    return best_model,best_score  

    

