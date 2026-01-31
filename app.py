# Importing necessary libraries
from flask import Flask, render_template, request, url_for, flash, redirect,session
import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
import mysql.connector
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import pymysql
pymysql.install_as_MySQLdb()
global ac_et1,ac_dt1,ac_svc1,ac_knn1
db=mysql.connector.connect(user="root",password="",port='3306',database='cyber_hacking')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/drug')
def drug():
    return render_template('drug.html')
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from cyber where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("load.html",myname=data[0][0])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        address = request.form['address']
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from cyber where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into cyber(Name,Email,Password,Age,Address,Contact)values(%s,%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,address,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')


@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df,output
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df.replace({'hacked':1,'poor security':0,'lost / stolen media':0,'inside job':0,'accidentally published':0,'lost / stolen computer':0,'unknown':0,'poor security/inside job':0,'improper setting, hacked':0,
            'accidentally uploaded':0,'Poor security':0,'poor security / hacked':1,'unprotected api':0,'publicly accessible Amazon Web Services (AWS) server':0,'unsecured S3 bucket':0,'inside job, hacked':1,
            'rogue contractor':0,'accidentally exposed ':0,'intentionally lost':0,'data exposed by misconfiguration':0,'misconfiguration/poor security':0,'social engineering':0,'accidentally exposed':0},inplace=True)
        df['Entity'] = le.fit_transform(df['Entity'])
        df['Organization type'] = le.fit_transform(df['Organization type'])


       # Assigning the value of x and y 
        x = df.drop(['Method'],axis=1)
        y = df['Method']
        # p_true =0.5
        # output = p_true>=np.random.rand()

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        global ac_et1,ac_dt1,ac_svc1,ac_knn1
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

            rf = RandomForestClassifier()
            rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            ac_et = accuracy_score(y_test, y_pred)
            ac_et1 = ac_et * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Random Forest Classifier is  ' + str(ac_et1) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            classifier = DecisionTreeClassifier(max_leaf_nodes=39, random_state=0)
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            ac_dt = accuracy_score(y_test, y_pred)
            ac_dt1 = ac_dt * 100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(ac_dt1) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            from sklearn.svm import SVC
            svc=SVC()
            svc=svc.fit(x_train,y_train)
            y_pred  =  svc.predict(x_test)            
            ac_svc = accuracy_score(y_test, y_pred)
            ac_svc1 = ac_svc * 100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(ac_svc1) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            
            from sklearn.ensemble import ExtraTreesClassifier
            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

            # Create the ExtraTreesClassifier model
            ex = ExtraTreesClassifier()
            # Perform GridSearchCV with 5-fold cross-validation
            grid_search = GridSearchCV(ex, param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            # Get the best model and predict on the test data
            best_ex = grid_search.best_estimator_
            y_pred = best_ex.predict(x_test)
            # Calculate accuracy
            ac_ex = accuracy_score(y_test, y_pred)
            ac_ex1 = ac_ex * 100
            msg = 'The accuracy obtained by ExtraTreeClassifier is ' + str(ac_ex1) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:

            cb = CatBoostClassifier()
            cb.fit(x_train, y_train)
            y_pred  = cb.predict(x_test)
            
            ac_knn = accuracy_score(y_test, y_pred)
            ac_knn1 = ac_knn * 100

            #GridSearchCV
            grid = {'max_depth': [3,4,5],'n_estimators':[100, 200, 300]}
            gscv = GridSearchCV (estimator = cb, param_grid = grid, scoring ='accuracy', cv = 5)
            gscv.fit(x,y)
            cbpg = gscv.predict(x_test)
            cbas= accuracy_score(y_test,cbpg)*100

            msg = 'The accuracy obtained by Cat Boost Classifier is ' + str(cbas) + str('%')
            return render_template('model.html', msg=msg)


    return render_template('model.html')


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
   
        f1 = int(request.form['Entity'])
        f2 = int(request.form['Year'])
        f3 = int(request.form['Records'])
        f4 = int(request.form['Organization type'])
        


        li = [f1,f2,f3,f4]
        print(li)
        

        cb = CatBoostClassifier()
        cb.fit(x_train, y_train)
        grid = {'max_depth': [3,4,5],'n_estimators':[100, 200, 300]}
        gscv = GridSearchCV (estimator = cb, param_grid = grid, scoring ='accuracy', cv = 5)
        gscv.fit(x,y)
        output = gscv.predict([li])
        print(output)
        print('result is ',output)

        if output == 0:
            msg = 'There is a CYBER HACKING '
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'There is No CYBER HACKING '
            return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')


@app.route('/graph')
def graph():
    global ac_et1,ac_dt1,ac_svc1,ac_knn1
    i = [ac_et1,ac_dt1,ac_svc1,ac_knn1]
    return render_template('graph.html',i=i)

if __name__=='__main__':
    app.run(debug=True)