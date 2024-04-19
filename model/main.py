import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data(): 
    data = pd.read_csv("data/data.csv")
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data.diagnosis = data.diagnosis.map({'M':1, 'B':0})
    return data 

def create_model(data): 
    scaler = StandardScaler()
    X = data.drop(['diagnosis'], axis=1)

    y = data.diagnosis 
    X = scaler.fit_transform(X) 

    # split the data 

    X_train, X_test, y_train, y_test = train_test_split (X, y, 
                                                         test_size=0.2, 
                                                         random_state=36)
    
    # train the model 
    logmodel = LogisticRegression() 
    logmodel.fit(X_train,y_train) 
    
    # test the model 
    y_pred = logmodel.predict(X_test) 
    print('model accuracy:', accuracy_score(y_test, y_pred)) 
    print("Classification report: \n", classification_report(y_test, y_pred))

    return logmodel, scaler   

def main(): 
    data = get_clean_data() 
    model, scaler  = create_model(data)
    with open('model/model.pkl', 'wb') as f: 
        pickle.dump(model,f)    
    with open('model/scaler.pkl', 'wb') as f: 
        pickle.dump(scaler,f)
    

if __name__ == '__main__': 
    main() 