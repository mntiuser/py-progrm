##linear regression
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('D:\\datasets 5thsem\\dataset\\Salary_Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)
x_pred
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,x_pred,color='red')
plt.title('salary vs Experience(training dataset)')
plt.xlabel('years of experience')
plt.ylabel('salary(in rupees)')
plt.show()
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,x_pred,color='green')
plt.title('salary vs Experience(test dataset)')
plt.xlabel('years of experience')
plt.ylabel('salary(in rupees)')
plt.show()

##Hold method
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data, iris.target, test_size=0.2, 
random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, 
random_state=42) 
clf=LogisticRegression().fit(x_train, y_train)  
score=clf.score(x_val, y_val) 
print (f"Validation accuracy: {score:.2f}") 
score_s=clf.score(x_test, y_test) 
print(f"Test accuracy: {score:.2f}")