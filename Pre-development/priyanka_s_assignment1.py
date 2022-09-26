# -*- coding: utf-8 -*-
"""Priyanka S_Assignment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i2RzcJRJCSxsRQwFYR02qAOHLuU0lMYI

# Basic Python

## 1. Split this string
"""

s = "Hi there Sam!"

a=s.split()
print(a)

"""*`italicized text`*## 2. Use .format() to print the following string. 

### Output should be: The diameter of Earth is 12742 kilometers.
"""

planet = "Earth"
diameter = 12742

print("The diameter of {} is {} kilometers".format("Earth",12742) )

"""## 3. In this nest dictionary grab the word "hello"
"""

d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}

x=d['k1'][3]['tricky'][3]['target'][3]
print(x)

"""# Numpy"""

import numpy as np

"""## 4.1 Create an array of 10 zeros? 
## 4.2 Create an array of 10 fives?
"""

array=np.zeros(10)
print(array)

array=np.ones(10)*5
print(array)



"""## 5. Create an array of all the even integers from 20 to 35"""

array=np.arange(20,36,2)
print(array)

"""## 6. Create a 3x3 matrix with values ranging from 0 to 8"""

x=np.arange(0,9).reshape(3,3)
print(x)

"""## 7. Concatinate a and b 
## a = np.array([1, 2, 3]), b = np.array([4, 5, 6])
"""

a=np.array([1,2,3])
b=np.array([4,5,6])
x=np.concatenate((a,b),axis=0)
print(x)

"""# Pandas

## 8. Create a dataframe with 3 rows and 2 columns
"""

import pandas as pd

data=[1,2,3]
x=pd.DataFrame(data,columns=['Numbers'])
print(x)

"""## 9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023"""

d=pd.date_range(start='1-1-2023',end='10-02-2023')
print(d)

"""## 10. Create 2D list to DataFrame

lists = [[1, 'aaa', 22],
         [2, 'bbb', 25],
         [3, 'ccc', 24]]
"""

lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]

d=pd.DataFrame(list,columns={'S.No','XXX','No'})