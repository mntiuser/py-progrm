from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#loading iris datas set
iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data, iris.target, test_size=0.2, random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
clf=LogisticRegression().fit(x_train, y_train)
score=clf.score(x_val, y_val)
print (f"Validation accuracy: {score:.2f}") 
score=clf.score(x_test, y_test)
print(f"Test accuracy: {score:.2f}")