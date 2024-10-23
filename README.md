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

#svm.1

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
cancer_data=datasets.load_breast_cancer()
cancer_data
x_train,x_test,y_train,y_test=train_test_split(cancer_data.data,cancer_data.target,test_size=0.3,random_state=402)
cls=svm.SVC()
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
y_test
pred
pred.shape
print("accuracy:",metrics.accuracy_score(y_test,y_pred=pred))
print("precision:",metrics.precision_score(y_test,y_pred=pred))
print("recall:",metrics.recall_score(y_test,y_pred=pred))
#get support vector indices
support_vector_indices=cls.support_
print(support_vector_indices)
#get number of support vectors per class
support_vectors_per_class=cls.n_support_
print(support_vectors_per_class)
#2
from sklearn.datasets import load_wine
wine_data=load_wine()
wine_data
x_train,x_test,y_train,y_test=train_test_split(wine_data.data,wine_data.target,test_size=0.3,random_state=402)
cls=svm.SVC(kernel="linear")
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
y_test
pred
pred.shape
print("accuracy:",metrics.accuracy_score(y_test,y_pred=pred))
print("precision:",metrics.precision_score(y_test,y_pred=pred,average='macro'))
print("Recall:",metrics.recall_score(y_test,y_pred=pred,average='macro'))
support_vectors_class=cls.support_
print(support_vectors_class)
support_vectors_per_class=cls.n_support_
print(support_vectors_per_class)
