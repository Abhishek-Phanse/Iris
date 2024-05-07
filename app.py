from flask import Flask, render_template, request, redirect, url_for
import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target

    
    slen = float(request.form['slen'])
    swid = float(request.form['swid'])
    plen = float(request.form['plen'])
    pwid = float(request.form['pwid'])
    temp = np.array([[slen, swid, plen,pwid]])
    new_test = pd.DataFrame(data = temp, columns = iris_data['feature_names'])

    iris_df = pd.DataFrame(data = iris_data['data'], columns = iris_data['feature_names'])
    iris_df['Iris type'] = iris_data['target']
    def f(x):
        if x == 0:
            val = 'setosa'
        elif x == 1:
            val = 'versicolor'
        else:
            val = 'virginica'
        return val
    
    iris_df['Iris_name'] = iris_df['Iris type'].apply(f)

    # KNN
    
    X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
    y = iris_df['Iris_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 35)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred1 = knn.predict(new_test)

    #LR

    X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
    y = iris_df['Iris_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state = 45)
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(max_iter=1000)
    LR.fit(X_train, y_train)
    y_pred2 = LR.predict(new_test)

    #DT

    X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
    y = iris_df['Iris_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state = 60)
    from sklearn.tree import DecisionTreeClassifier
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    y_pred3 = DT.predict(new_test)

    return render_template('result.html', output1=y_pred1[0], output2=y_pred2[0],  output3=y_pred3[0])

if __name__ == '__main__':
    app.run(debug=True)
