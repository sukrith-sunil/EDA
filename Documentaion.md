<a name="br1"></a> 

ML-MAJOR**-AUGUS**T

ML-08-SPB2

Submitted On:

5/10/2022

Submitted By:

Sukrith Sunil

Amrita School Of Engineering

Amritapuri

1



<a name="br2"></a> 

**INRODUCTION**

In this project we perform,perform EDA(Exploratory

Data Analysis) and apply a

suitable Classifier,Regressor or Clusterer and calculate

the accuracy of the model on a data set CARS.csv

2



<a name="br3"></a> 

**FUNCTIONS**

dataorg()

graphs()

acc\_check()

CSV FILE

https://drive.google.com/file/d/1-AyVrZz6vJtlq2f\_dGn0tFjFBCpI8Cf6/view?

usp=sharing

3



<a name="br4"></a> 

**SOURCE CODE**

import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

import statsmodels as sm

from statsmodels.stats.outliers\_influence import variance\_inflation\_factor

from sklearn.model\_selection import

train\_test\_split,GridSearchCV,RandomizedSearchCV

from sklearn.linear\_model import LinearRegression,Ridge,Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import

RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import r2\_score,mean\_squared\_error

from sklearn import preprocessing

df\_cars = pd.read\_csv("CARS.csv")

def dataorg():

df\_cars.horsepower =

df\_cars.horsepower.str.replace('?','NaN').astype(float)

df\_cars.horsepower.fillna(df\_cars.horsepower.mean(),inplace=True)

df\_cars.horsepower = df\_cars.horsepower.astype(int)

df\_cars.info()

df\_cars['car name'] = df\_cars['car name'].str.replace('chevroelt|chevrolet|

chevy','chevrolet')

df\_cars['car name'] = df\_cars['car name'].str.replace('maxda|

mazda','mazda')

df\_cars['car name'] = df\_cars['car name'].str.replace('mercedes|mercedes-

benz|mercedes benz','mercedes')

df\_cars['car name'] = df\_cars['car name'].str.replace('toyota|

toyouta','toyota')

df\_cars['car name'] = df\_cars['car name'].str.replace('vokswagen|

volkswagen|vw','volkswagen')

df\_cars.groupby(['car name']).sum().head()

display(df\_cars.describe().round(2))

df\_cars['origin'] = df\_cars['origin'].replace({1: 'USA', 2: 'Europe', 3: 'Asia'})

df\_cars.head()

4



<a name="br5"></a> 

def graphs():

sns\_plot = sns.histplot(df\_cars["mpg"], color="red", label="100% Equities", kde=True,

stat="density", linewidth=0)

plt.figure(figsize=(10,6))

sns.heatmap(df\_cars.corr(),cmap=plt.cm.Reds,annot=True)

plt.title('Heatmap displaying the relationship between the features of the data',

fontsize=13)

plt.show()

fig, ax = plt.subplots(figsize = (5, 5))

sns.countplot(x = df\_cars.origin.values, data=df\_cars)

labels = [item.get\_text() for item in ax.get\_xticklabels()]

labels[0] = 'America'

labels[1] = 'Europe'

labels[2] = 'Asia'

ax.set\_xtickladef graphs():

sns\_plot = sns.histplot(df\_cars["mpg"], color="red", label="100% Equities", kde=True,

stat="density", linewidth=0)

bels(labels)

ax.set\_title("Cars manufactured by Countries")

plt.show()

fig, ax = plt.subplots(6, 2, figsize = (15, 13))

sns.boxplot(x= df\_cars["mpg"], ax = ax[0,0])

sns.histplot(df\_cars['mpg'], ax = ax[0,1])

sns.boxplot(x= df\_cars["cylinders"], ax = ax[1,0])

sns.histplot(df\_cars['cylinders'], ax = ax[1,1])

sns.boxplot(x= df\_cars["displacement"], ax = ax[2,0])

sns.histplot(df\_cars['displacement'], ax = ax[2,1])

sns.boxplot(x= df\_cars["horsepower"], ax = ax[3,0])

sns.histplot(df\_cars['horsepower'], ax = ax[3,1])

sns.boxplot(x= df\_cars["weight"], ax = ax[4,0])

sns.histplot(df\_cars['weight'], ax = ax[4,1])

sns.boxplot(x= df\_cars["acceleration"], ax = ax[5,0])

sns.histplot(df\_cars['acceleration'], ax = ax[5,1])

plt.tight\_layout()

5



<a name="br6"></a> 

plt.figure(1)

f,axarr = plt.subplots(4,2, figsize=(10,10))

mpgval = df\_cars.mpg.values

axarr[0,0].scatter(df\_cars.cylinders.values, mpgval)

axarr[0,0].set\_title('Cylinders')

axarr[0,1].scatter(df\_cars.displacement.values, mpgval)

axarr[0,1].set\_title('Displacement')

axarr[1,0].scatter(df\_cars.horsepower.values, mpgval)

axarr[1,0].set\_title('Horsepower')

axarr[1,1].scatter(df\_cars.weight.values, mpgval)

axarr[1,1].set\_title('Weight')

axarr[2,0].scatter(df\_cars.acceleration.values, mpgval)

axarr[2,0].set\_title('Acceleration')

axarr[2,1].scatter(df\_cars["model year"].values, mpgval)

axarr[2,1].set\_title('Model Year')

axarr[3,0].scatter(df\_cars.origin.values, mpgval)

axarr[3,0].set\_title('Country Mpg')

axarr[3,0].set\_xticks([1,2,3])

axarr[3,0].set\_xticklabels(["USA","Europe","Asia"])

axarr[3,1].axis("off")

f.text(-0.01, 0.5, 'Mpg', va='center', rotation='vertical', fontsize = 12)

plt.tight\_layout()

plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})

cData\_attr = df\_cars.iloc[:, 0:7]

sns.pairplot(cData\_attr, diag\_kind='kde')

df\_cars.hist(figsize=(12,8),bins=20)

plt.show()

def acc\_check():

#accuracy check

feature\_cols = ['mpg','displacement','horsepower','weight','acceleration']

X = df\_cars[feature\_cols]

y = df\_cars.cylinders

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, random\_state=0)

from sklearn.linear\_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X\_train, y\_train)

y\_pred\_class = logreg.predict(X\_test)

from sklearn import metrics

print('\n\n\t\tAccuracy = ',(metrics.accuracy\_score(y\_test, y\_pred\_class))\*100,'%')

\# main pg

dataorg()

graphs()

acc\_check()



<a name="br7"></a> 

**SCREENSHOTS**

1



<a name="br8"></a> 

**SCREENSHOTS**

1



<a name="br9"></a> 

**SCREENSHOTS**

1

