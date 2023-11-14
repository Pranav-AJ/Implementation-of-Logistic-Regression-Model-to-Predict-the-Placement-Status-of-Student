# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3. Import LabelEncoder and encode the dataset.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7. Find new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.J.PRANAV
RegisterNumber: 212222230107
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data_Full_Class.csv")
df
```
```
df.head()
```
```
df.tail()
```
```
df.info()
```
```
df=df.drop('sl_no',axis=1)
df
```
```
df=df.drop(['ssc_b','hsc_b','gender'],axis=1)
df
```
```
df.shape
```

**Data Encoding**
```
df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
```
```
df.info()
```

```
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
```

```
df.info()
```
```
df
```
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(Y)
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0]])
```
## Output:

### Placement data

![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/efa2a366-e187-48fc-86d2-17145a531caf)

### df.head()
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/76efaf22-2d85-4b59-b765-92f1e19a0655)

### df.tail()
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/ff835b82-adcc-4f73-b10d-57662a25c23d)

### df.info()
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/3611bbff-7e02-4f3f-840a-24348f36703d)

![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/9f5820ea-1584-44bd-b7b7-bdd31cfd7760)

![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/8574d812-121a-4c11-a6c1-c9a6ec51e6e1)

![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/dda11c23-739f-47c4-a83c-07ee9b349afb)
### df
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/2110cfab-820f-464f-9720-a0e33c3813c4)
### data status
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/fc3fc45c-6388-48a8-897e-d2e38030186a)
### y_prediction value
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/463d6303-35e8-46b1-a668-89606fafb623)
### accuracy value
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/14c2f0f0-696a-4318-bddc-9322c78d517b)
### confusion matrix
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/058aab62-799f-4275-9034-11651c323559)
### classification of report
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/0b52d699-fde9-467f-90e9-72739388eba1)

### prediction of LR
![image](https://github.com/Pranav-AJ/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118904526/5fa1b3f4-fed1-4de5-9b45-948bd1d65743)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
