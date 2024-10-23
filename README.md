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

#bootstrapping.1
import numpy as np
import scipy.stats as st
population_dist=np.random.poisson(20,size=1000)
population_dist
sample=np.random.choice(population_dist,10,replace=False)
bootstrap_dist=[(np.random.choice(sample,10,replace=True))]
sample
bootstrap_dist
bootstrap_mean=[np.mean(bootstrap_dist)]
bootstrap_mean
st.t.interval(confidence=0.95,df=9,loc=np.mean(bootstrap_dist))
#2.
import pandas as pd
df = pd.DataFrame({
    'Grade': students_cgpa
})
Df
df.describe()
import numpy as np
sample1 = np.random.choice(students_cgpa, 15, replace = False)
bootstrap_sample1 = [(np.random.choice(students_cgpa, 15, replace = True))]
sample1
bootstrap_sample1
import matplotlib
import matplotlib.pyplot as pt
bootstrap_100 = pd.DataFrame({'meangrade':[df.sample(20,replace=True).Grade.mean() for i in range (100)]})
bootstrap_100.meangrade.hist(histtype = 'step')
pt.axvline(df.Grade.mean(),color='red')
bootstrap_100.meangrade.quantile(0.025), bootstrap_100.meangrade.quantile(0.075)
yrs_of_exp = np.array([1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,
                       8.2,8.7,9,9.5,9.6,10.3,10.5])
salary = np.array([39343, 46205 ,43525 , 37731,  39891, 56642 , 60150 , 54445, 64445, 
                   57189 ,63218,  55794 , 56957, 57081, 6111, 67938 ,66029, 83088,
                   81363, 93940, 91738, 98273, 101302, 113812, 109431 ,105582, 116969,
                  112635, 122391, 121872])
bootstrap_sample3 = [np.random.choice(len(yrs_of_exp), size=10, replace=True) for i in range(3)]
bootstrap_sample3
#3.
from sklearn.linear_model import LinearRegression
def estimate_regression_coefficient(sample_idx):
    x_sample = yrs_of_exp[sample_idx].reshape(-1,1)
    y_sample = salary[sample_idx]
    model = LinearRegression()
    model.fit(x_sample, y_sample)
    return model.coef_[0]
coefficients = [estimate_regression_coefficient(sample_idx) for sample_idx in bootstrap_sample3]
print(f"Bootstrap Regression Coefficients: {coefficients}")
Bootstrap Regression Coefficients: [9966.853434674646, 8967.275729678398, 9942.471069839978]
average_coef = np.mean(coefficients)
print(average_coef)


#EM.2
import pandas as pd
df=pd.read_csv("Skulls.csv")
df.head(5)
import seaborn as sns In [187]: x_train,x_test=train_test_split(df,test_size=0.3,random_state=42) x_train
df.drop(['rownames'],axis='columns',inplace=True) 
df
df.drop(['epoch'],axis='columns',inplace=True)
df
import matplotlib.pyplot as plt
plt.scatter(x_train['mb'],x_train['bh']); plt.title("Scatter plot between mb and bh column") plt.xlabel("maximum breadth") plt.ylabel("basi bregmatic heights")…..
from sklearn.mixture import GaussianMixture In [234]: model=GaussianMixture(n_components=3) model.fit(x_train,None)
model.fit_predict(x_test)
a=model.fit_predict(x_train)
#1.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris=datasets.load_iris()
iris
import seaborn as sns
x_train,x_test=train_test_split(iris.data,test_size=0.3,random_state=42
x_train
import matplotlib.pyplot as plt
plt.scatter(x_train[:,0],x_train[:,1]); plt.title("Scatter plot between 0th and 1st column")
from sklearn.mixture import GaussianMixture
model=GaussianMixture(n_components=3) model.fit(x_train,None)

#KNN.1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
dataset=pd.read_csv('KNN_Dataset.csv')
dataset.head()
# replace zeroes
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.NAN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NAN,mean)
dataset.head()
x=dataset.drop('Outcome',axis=1)
y=dataset['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
768**(1/2)
#feature scaling
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
x_train
classifier = KNeighborsClassifier(n_neighbors=28, p=2, metric="euclidean")
classifier.fit(x_train,y_train)
y_pred =classifier.predict(x_test)
y_pred
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))






