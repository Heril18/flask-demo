import numpy as np
import pickle
import matplotlib as plt
import pandas as pd
df=pd.read_csv("hiring.csv")
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
x=df.iloc[:,:4]
y=df.iloc[:,-1]
inttoword= {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, '0': 0}
i=0
e=[]
for j in x['experience']:
 j=inttoword[str(j)]
 e.append(j)
x['exp']=e
x.drop('experience',axis=1,inplace=True)
x.drop('Index',axis=1,inplace=True)
print(x)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

print(model.predict([[2,9,6]]))

