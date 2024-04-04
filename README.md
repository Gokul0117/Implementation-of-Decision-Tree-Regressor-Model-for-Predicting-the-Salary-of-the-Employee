# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Gokul J
RegisterNumber:  212222230038
*/
```
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()
data.isnull().sum

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[['Position','Level']]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
## Head:

![318891685-6b9805cf-30b6-49e4-9ee5-cfc70c9898ea](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121165938/66b33c25-5974-4be0-91a4-ddc8756fa152)

## MSE:

![318891906-aaa69afd-0a05-41f6-aefa-7d61d8a9ea54](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121165938/72548600-b2a5-4a37-9910-dcb4dbfecdae)
## R2 :

![318892072-734caf77-f008-4a2d-a625-a88702f7460f](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121165938/cd8db09c-d42f-44fe-bb66-08b0fbd0a90d)

## DATA PREDICT :

![318892296-c02c51d8-1fbe-4824-849a-28b3143e1a6c](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121165938/a3c080a2-26f7-44df-b7d3-a1b3d0663eab)

## DECISION TREE :
![318894601-3e22738b-6296-4606-a21f-db863cb00833](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121165938/5833d787-b35a-47ab-ac52-85c444f5304d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
