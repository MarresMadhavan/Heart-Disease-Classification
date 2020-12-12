import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
ds = pd.read_csv('/Users/marreswaran/Desktop/MY PERSONAL/MACHINE LEARNING/LECTURES RAJKUMAR BHUNIA/CODE FILES AND DATA SETS/Project-3/heart.csv')
print(ds)
print(ds.isna().sum())
print(ds.head())
x = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values
print(x)
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2,5,6,8,10,11,12])],remainder='passthrough')
x = ct.fit_transform(x)
sc = StandardScaler()
x = sc.fit_transform(x)
print (x)
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.25,random_state=4)
classifier = SVC(kernel='linear',random_state=1)
classifier.fit(x_tr,y_tr)
ypred= classifier.predict(x_te)
cm=confusion_matrix(y_te,ypred)
acc = accuracy_score(y_te,ypred)
print(cm)
print(acc)
plot_confusion_matrix(cm)
plt.show()
