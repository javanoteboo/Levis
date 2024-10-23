#bayesian 2
import pandas as pd
df=pd.read_csv("Pima.tr.csv")
df.head(10)
df.drop(['rownames'],axis='columns',inplace=True)
df
inputs=df.drop('type',axis=1)
target=df['type']
inputs
inputs.columns[inputs.isna().any()]
inputs.age=inputs.age.fillna(inputs.age.mean())
inputs.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3,random_state=42)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)
x_test[0:10]
y_test[0:10]
x_test
model.predict(x_test[0:10])
model.predict_proba(x_test[:10])

#2.
import pandas as pd
df= pd.read_csv("TitanicSurvival.csv")
df.head(10)
df.drop(['rownames'],axis='columns',inplace=True)
df
inputs=df.drop("survived",axis=1)
target=df["survived"]
inputs
target
dummies=pd.get_dummies(inputs.sex)
dummies.head(3)
inputs=pd.concat([inputs,dummies],axis="columns")
inputs.head(3)
inputs.drop(["sex","male"],axis="columns",inplace=True)
inputs.head(3)
inputs.columns[inputs.isna().any()]
inputs.age=inputs.age.fillna(inputs.age.mean())
inputs.head()
inputs.replace(to_replace="1st",value="1",inplace=True)
inputs.replace(to_replace="2nd",value="2",inplace=True)
inputs.replace(to_replace="3rd",value="3",inplace=True)
inputs
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3,random_state=42)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)
x_test[0:10]
y_test[0:10]
import numpy as np
b=np.array([[14,3,0]])
x_test
model.predict(x_test[0:10])
model.predict_proba(x_test[:10])
import numpy as np
b=np.array([[60,0,3]])
model.predict(b)
b=np.array([[35,0,2]])
model.predict(b)
b=np.array([[15,1,2]])
model.predict(b)
b=np.array([[18,1,1]])
model.predict(b)
model.predict_proba(b)
