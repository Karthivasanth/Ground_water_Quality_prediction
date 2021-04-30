
import os
os.chdir(r'C:\Users\ASUS-PC\Documents\Python op')

import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('water_data.csv',encoding="ISO-8859-1")
data.fillna(0, inplace=True)
data.head()

data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')
data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
data['PH']=pd.to_numeric(data['PH'],errors='coerce')
data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')


start=2
end=1992
station=data.iloc [start:end ,0]
location=data.iloc [start:end ,1]
state=data.iloc [start:end ,2]
do= data.iloc [start:end ,4].astype(np.float64)
value=0
ph = data.iloc[ start:end,5]  
co = data.iloc [start:end ,6].astype(np.float64)   
year=data.iloc[start:end,11]
tc=data.iloc [2:end ,10].astype(np.float64)
bod = data.iloc [start:end ,7].astype(np.float64)
na= data.iloc [start:end ,8].astype(np.float64)
na.dtype

data.head()

data=pd.concat([station,location,state,do,ph,co,bod,na,tc,year],axis=1)
data. columns = ['station','location','state','do','ph','co','bod','na','tc','year']

data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))

data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))

data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))

data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))

data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))

data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))

data.head()
data.dtypes

data['wph']=data.npH * 0.165
data['wdo']=data.ndo * 0.281
data['wbdo']=data.nbdo * 0.234
data['wec']=data.nec* 0.009
data['wna']=data.nna * 0.028
data['wco']=data.nco * 0.281
data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco 
data

ag=data.groupby('year')['wqi'].mean()

ag.head()

data=ag.reset_index(level=0,inplace=False)
data

year=data['year'].values
AQI=data['wqi'].values
data['wqi']=pd.to_numeric(data['wqi'],errors='coerce')
data['year']=pd.to_numeric(data['year'],errors='coerce')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(year,AQI, color='red')
plt.show()
data

data = data[np.isfinite(data['wqi'])]
data.head()

cols =['year']
y = data['wqi']
x=data[cols]

plt.scatter(x,y)
plt.show()

import matplotlib.pyplot as plt
data=data.set_index('year')
data.plot(figsize=(15,6))
plt.show()

from sklearn import neighbors,datasets
data=data.reset_index(level=0,inplace=False)
data

from sklearn import linear_model
from sklearn.model_selection import train_test_split

cols =['year']

y = data['wqi']
x=data[cols]
reg=linear_model.LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
reg.fit(x_train,y_train)
a=reg.predict(x_test)
a

y_test

from sklearn.metrics import mean_squared_error
print('mse:%.2f'%mean_squared_error(y_test,a))

dt = pd.DataFrame({'Actual': y_test, 'Predicted': a}) 

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]
x

alpha = 0.1 #Step size
iterations = 3000 #No. of iterations
m = y.size #No. of data points
np.random.seed(4) #Setting the seed
theta = np.random.rand(2) #Picking some random values to start with

def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]

#Print the results...
print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()


import numpy as np
newB=[74.92, 4.24]

def rmse(y,y_pred):
    rmse= np.sqrt(sum(y-y_pred))
    return rmse
   

y_pred=x.dot(newB)

dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  
dt=pd.concat([data, dt], axis=1)
dt

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))

x_axis=dt.year
y_axis=dt.Actual
y1_axis=dt.Predicted
plt.scatter(x_axis,y_axis)
plt.plot(x_axis,y1_axis,color='r')
plt.title("linear regression")

plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,r2_score,mean_gamma_deviance,explained_variance_score,max_error

print("  ")
print("Linear Regression:")
print("R2 Score:",r2_score(y,y_pred))
print("Root Mean Sqaure:",np.sqrt(mean_squared_error(y,y_pred)))
print("Explained Variance Score:",explained_variance_score(y,y_pred))
print("Max Error:",max_error(y,y_pred))
print("Mean Gamma Devience:",mean_gamma_deviance(y,y_pred))
print("---------------------------------------------------------------------")
print("  ")


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
b = pol_reg.predict(poly_reg.fit_transform(x_test))
print("Polynomial Regression:")
print("R2 Score:",r2_score(y_test,b))
print("Root Mean Sqaure:",np.sqrt(mean_squared_error(y_test,b)))
print("Explained Variance Score:",explained_variance_score(y_test,b))
print("Max Error:",max_error(y_test,b))
print("Mean Gamma Devience:",mean_gamma_deviance(y_test,b))
print("---------------------------------------------------------------------")
print("  ")



from sklearn.ensemble import RandomForestRegressor
regres = RandomForestRegressor(min_samples_split=75,min_samples_leaf=10)
hell = regres.fit(x_train,y_train)
c = regres.predict(x_test)
print("Random Forest Regression:")
print("R2 Score:",r2_score(y_test,c))
print("Root Mean Sqaure:",np.sqrt(mean_squared_error(y_test,c)))
print("Explained Variance Score:",explained_variance_score(y_test,c))
print("Max Error:",max_error(y_test,c))
print("Mean Gamma Devience:",mean_gamma_deviance(y_test,c))
print("---------------------------------------------------------------------")
print("  ")


from sklearn.linear_model import Lasso
lasso_reg = Lasso(normalize=True)
lasso_reg.fit(x_train,y_train)
d = lasso_reg.predict(x_test)
print("Lasso Regression:")
print("R2 Score:",r2_score(y_test,d))
print("Root Mean Sqaure:",np.sqrt(mean_squared_error(y_test,d)))
print("Explained Variance Score:",explained_variance_score(y_test,d))
print("Max Error:",max_error(y_test,d))
print("Mean Gamma Devience:",mean_gamma_deviance(y_test,d))
print("---------------------------------------------------------------------")
print("  ")


import scipy.stats as stats
import math
sample = np.random.choice(a=y_pred,size = 11)
sample_mean = sample.mean()
z_critical = stats.norm.ppf(q = 0.95)
pop_stdev = y_pred.std()
margin_of_error = z_critical * (pop_stdev/math.sqrt(250))
confidence_interval = (sample_mean - margin_of_error,sample_mean +
margin_of_error)
print("Confidence interval:",end=" ")
print(confidence_interval)