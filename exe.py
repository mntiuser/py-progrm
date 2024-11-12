import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
iris=pd.read_csv('D:\\datasets 5thsem\\dataset\\Iris1.csv')
#print(iris)
fig=iris[iris.variety=='Setosa'].plot.scatter(x='petal.length',y='petal.width',color='orange',label='setosa')
fig=iris[iris.variety=='Versicolor'].plot.scatter(x='petal.length',y='petal.width',color='blue',label='versicolor',ax=fig)
fig=iris[iris.variety=='Virginica'].plot.scatter(x='petal.length',y='petal.width',color='green',label='virginica',ax=fig)
fig.set_xlabel('petal.length')
fig.set_xlabel('petal.width')
fig.set_title('petal length and width')
fig=plt.gcf()
fig.set_size_inches(6,6)
#plt.show()
#display all data whose values==1.4
p_w=iris[iris['petal.length']==1.4]
#print("displaying all the dat whose value is 1.4:",p_w)
##how many times each value in petal length is given
co=iris.groupby('petal.length')['petal.length'].count()
#print("count of petal length:",co)
##to find the sum of petal length
s=iris['petal.length'].sum()
#print("sum of petal length:",s)
##maximum
mm=iris['petal.length'].max()
#print("maximum:",mm)
#add 1,0to all the inputs
add=iris['sepal.length']+1.0
#print("adding 1.0:",add)

##linear regression
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

##plynomial regression
data_set=pd.read_csv('D:\\datasets 5thsem\\dataset\\Salary_Data.csv')
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:, 1].values
from sklearn.linear_model import LinearRegression
lin_regs=LinearRegression()
lin_regs.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
poly_regs=PolynomialFeatures (degree=2)
x_poly=poly_regs.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)
plt.scatter(x,y,color="blue")
plt.plot(x,lin_regs.predict(x),color="red")
plt.title("Bluff detection model(Linear Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()
plt.scatter(x,y,color="blue")
plt.plot(x,lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")
plt.title("Bluff detection model (Polynomial Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show() 

##HOLD OUT METHOD
iris_data=pd.read_csv('D:\\datasets 5thsem\\dataset\\Iris.csv')
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
# Split dataset into training, validation, and test sets 
iris=load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, 
random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, 
random_state=42) 
# Train logistic regression model on training set 
clf = LogisticRegression().fit(x_train, y_train) 
# Evaluate model on validation set 
score = clf.score(x_val, y_val) 
print (f"Validation accuracy: {score:.2f}") 
# Evaluate model on test set 
score = clf.score(x_test, y_test) 
print(f"Test accuracy: {score:.2f}")

##LEAVEONEOUT CROSS-VALIDATION
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import accuracy_score 
# Load Iris dataset 
iris=load_iris()
# Define leave-one-out cross-validation object 
loo=LeaveOneOut() 
# Train logistic regression model using leave-one-out cross-validation 
scores=[] 
for train_index,test_index in loo.split(iris.data): 
    x_train,x_test=iris.data[train_index],iris.data[test_index] 
    y_train,y_test=iris.target[train_index],iris.target[test_index] 
clf=LogisticRegression().fit(x_train, y_train) 
y_pred=clf.predict(x_test) 
scores.append(accuracy_score(y_test,y_pred)) 
# Compute average accuracy across all samples 
score_a=sum(scores)/len(scores) 
print(f"Accuracy: {score:.2f}")