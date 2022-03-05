# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:13:05 2021

@author: Jovan
"""

import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels .api as sm
import math

#from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



#функција имплементирана на вежбама, служи ради описивања исправности модела
def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(y_test, y_predicted) # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))


###################


# Учитавање података
data=pd.read_csv("dataset.csv")


# Уклањање мерења која се односе на друга места 
data.drop('PM_Dongsi', axis='columns',inplace=True)
data.drop('PM_Dongsihuan', axis='columns',inplace=True)
data.drop('PM_Nongzhanguan', axis='columns',inplace=True)
#

# Подела почетног скупа података на подкатегорије
#1 годишње доба
summer=data[data['season']==2]
winter=data.loc[data['season']==4]
spring=data.loc[data['season']==1]
autumn=data.loc[data['season']==3]


#2 део дана преподне послеподне вече
prepodne=[4,5,6,7,8,9,10,11]
poslepodne=[12,13,14,15,16,17,18,19]
noc=[20,21,22,23,0,1,2,3]

day=data[data['hour'].isin(poslepodne) ]
mourning=data[data['hour'].isin(prepodne)]
night=data[data['hour'].isin(noc)]

#3 правци ветрова

nw=data[data['cbwd']=='NW']
ne=data[data['cbwd']=='NE']
sw=data[data['cbwd']=='SW']
se=data[data['cbwd']=='SE']
cv=data[data['cbwd']=='cv']

#Груписање истих уз претпоставку да ће средње вредности бити приближније 
#у зависности од годишњег доба и дела дана

summer_day=summer[summer['hour'].isin(poslepodne)]
summer_mourning=summer[summer['hour'].isin(prepodne)]
summer_night=summer[summer['hour'].isin(noc)]

autumn_day=autumn[autumn['hour'].isin(poslepodne)]
autumn_mourning=autumn[autumn['hour'].isin(prepodne)]
autumn_night=autumn[autumn['hour'].isin(noc)]

winter_day=winter[winter['hour'].isin(poslepodne)]
winter_mourning=winter[winter['hour'].isin(prepodne)]
winter_night=winter[winter['hour'].isin(noc)]

spring_day=spring[spring['hour'].isin(poslepodne)]
spring_mourning=spring[spring['hour'].isin(prepodne)]
spring_night=spring[spring['hour'].isin(noc)]



## ШИФРА СЕ САСТОЈИ ОД НАЗИВА ГОДИШЊЕГ ДОБА И ДЕЛА ДАНА
# SU-summer; SP-spring WI -winter; AU -autumn
# D-Day; M-Mourning; N-Night

#----------ПРОСЕЧНА ВРЕДНОСТ----------------------------

avg_tempSP=spring['TEMP'].mean()

avg_tempWI=winter['TEMP'].mean()

avg_tempSU=summer['TEMP'].mean()

avg_tempAU=autumn['TEMP'].mean()

avg_temp=data['TEMP'].mean()

avg_tempSUD=summer_day['TEMP'].mean()

avg_tempSUM=summer_mourning['TEMP'].mean()

avg_tempSUN=summer_night['TEMP'].mean()

avg_tempSPD=spring_day['TEMP'].mean()

avg_tempSPM=spring_mourning['TEMP'].mean()

avg_tempSPN=spring_night['TEMP'].mean()

avg_tempAUM=autumn_mourning['TEMP'].mean()

avg_tempAUD=autumn_day['TEMP'].mean()

avg_tempAUN=autumn_night['TEMP'].mean()

avg_tempWIN=winter_night['TEMP'].mean()

avg_tempWID=winter_day['TEMP'].mean()

avg_tempWIM=winter_mourning['TEMP'].mean()


avg_pm=data['PM_US Post'].mean()


avg_pmSP=spring['PM_US Post'].mean()
avg_pmWI=winter['PM_US Post'].mean()
avg_pmSU=summer['PM_US Post'].mean()
avg_pmAU=autumn['PM_US Post'].mean()

avg_pmSUD=summer_day['PM_US Post'].mean()

avg_pmSUM=summer_mourning['PM_US Post'].mean()

avg_pmSUN=summer_night['PM_US Post'].mean()

avg_pmSPD=spring_day['PM_US Post'].mean()

avg_pmSPM=spring_mourning['PM_US Post'].mean()

avg_pmSPN=spring_night['PM_US Post'].mean()

avg_pmAUM=autumn_mourning['PM_US Post'].mean()

avg_pmAUD=autumn_day['PM_US Post'].mean()

avg_pmAUN=autumn_night['PM_US Post'].mean()

avg_pmWIN=winter_night['PM_US Post'].mean()

avg_pmWID=winter_day['PM_US Post'].mean()

avg_pmWIM=winter_mourning['PM_US Post'].mean()

#---------------МЕДИЈАНА---------------------------------------


median_HUMI=data['HUMI'].median()

median_Iws = data['Iws'].median()

median_pm=data['PM_US Post'].median()

median_DEWP=data['DEWP'].median()

median_PRES=data['PRES'].median()


median_pmSP=spring['PM_US Post'].median()
median_pmWI=winter['PM_US Post'].median()
median_pmSU=summer['PM_US Post'].median()
median_pmAU=autumn['PM_US Post'].median()

median_pmSUD=summer_day['PM_US Post'].median()

median_pmSUM=summer_mourning['PM_US Post'].median()

median_pmSUN=summer_night['PM_US Post'].median()

median_pmSPD=spring_day['PM_US Post'].median()

median_pmSPM=spring_mourning['PM_US Post'].median()

median_pmSPN=spring_night['PM_US Post'].median()

median_pmAUM=autumn_mourning['PM_US Post'].median()

median_pmAUD=autumn_day['PM_US Post'].median()

median_pmAUN=autumn_night['PM_US Post'].median()

median_pmWIN=winter_night['PM_US Post'].median()

median_pmWID=winter_day['PM_US Post'].median()

median_pmWIM=winter_mourning['PM_US Post'].median()

#
median_tempSP=spring['TEMP'].median()
median_tempWI=winter['TEMP'].median()
median_tempSU=summer['TEMP'].median()
median_tempAU=autumn['TEMP'].median()

median_tempSUD=summer_day['TEMP'].median()

median_tempSUM=summer_mourning['TEMP'].median()

median_tempSUN=summer_night['TEMP'].median()

median_tempSPD=spring_day['TEMP'].median()

median_tempSPM=spring_mourning['TEMP'].median()

median_tempSPN=spring_night['TEMP'].median()

median_tempAUM=autumn_mourning['TEMP'].median()

median_tempAUD=autumn_day['TEMP'].median()

median_tempAUN=autumn_night['TEMP'].median()

median_tempWIN=winter_night['TEMP'].median()

median_tempWID=winter_day['TEMP'].median()

median_tempWIM=winter_mourning['TEMP'].median()


# Средње вредности концентрације пм честица ћу заменити медианом у зависности од категорије

summer_night=summer_night.fillna(value={'PM_US Post':median_pmSUN})
summer_mourning=summer_mourning.fillna(value={'PM_US Post':median_pmSUM})
summer_day=summer_day.fillna(value={'PM_US Post':median_pmSUD})



spring_mourning=spring_mourning.fillna(value={'PM_US Post':median_pmSPM})
spring_night=spring_night.fillna(value={'PM_US Post':median_pmSPN})
spring_day=spring_day.fillna(value={'PM_US Post':median_pmSPD})


autumn_mourning=autumn_mourning.fillna(value={'PM_US Post':median_pmAUM})
autumn_night=autumn_night.fillna(value={'PM_US Post':median_pmAUN})
autumn_day=autumn_day.fillna(value={'PM_US Post':median_pmAUD})


winter_mourning=winter_mourning.fillna(value={'PM_US Post':median_pmWIM})
winter_day=winter_day.fillna(value={'PM_US Post':median_pmWID})
winter_night=winter_night.fillna(value={'PM_US Post':median_pmWIN})







# In[20]:



#додајем недостајуће податке у табелу

spring=pd.concat([spring_mourning,spring_day,spring_night])
summer=pd.concat([summer_mourning,summer_day,summer_night])
autumn=pd.concat([autumn_mourning,autumn_day,autumn_night])
winter=pd.concat([winter_mourning,winter_day,winter_night])

data=pd.concat([spring,summer,autumn,winter])
# уклањање непоузданих редова

data=data.drop([51891,45922,47954,49271,51328,51277],axis='rows',inplace=False)

# замена непопуњених обележја

data=data.fillna(value={'precipitation':0,'Iprec':0})
data=data.fillna(value={'Iws':median_Iws,'Iprec':0})
data=data.fillna(value={'day':1,'Iprec':0})
data=data.fillna(value={'hour':1,'Iprec':0})
data=data.fillna(value={'season':1,'Iprec':0})
data=data.fillna(value={'DEWP':median_DEWP,'HUMI':median_HUMI,'PRES':median_PRES})

#претпоставио сам да је вредност за 'Iws'>120, међутим добијам прецизнији модел ако оставим нетакнуто то обележје 
#data["Iws"].values[data["Iws"] > 120] = median_Iws




#додајем поље које се односи на квалитет ваздуха у зависности од количине PM2.5 честица
kvalitet_vazduha=['Good','Moderate','Unhealty for sensitive groups', 'Unhealty', 'Very unhealty','Hazardous']
#1,2,3,4,5,6

data_GO=data.loc[(data['PM_US Post']>0.0) & (data['PM_US Post']<12.0)]
data_GO['AQ']=1

data_MO=data.loc[(data['PM_US Post']>12.1) & (data['PM_US Post']<35.4)]
data_MO['AQ']=2

data_US=data.loc[(data['PM_US Post']>35.5) & (data['PM_US Post']<55.4)]
data_US['AQ']=3

data_UN=data.loc[(data['PM_US Post']>55.5) & (data['PM_US Post']<150.4)]
data_UN['AQ']=4
 
data_VU=data.loc[(data['PM_US Post']>150.5) & (data['PM_US Post']<250.4)]
data_VU['AQ']=5

data_HA=data.loc[data['PM_US Post']>250.5]
data_HA['AQ']=6


dataAQ=pd.concat([data_HA,data_VU,data_GO,data_MO,data_UN,data_US])



#означавање нумеричких вредности за правац ветра

dataAQ1=dataAQ.loc[dataAQ['cbwd']=='NW']
dataAQ1['cbwd']=1
dataAQ2=dataAQ.loc[dataAQ['cbwd']=='NE']
dataAQ2['cbwd']=2
dataAQ3=dataAQ.loc[dataAQ['cbwd']=='SW']
dataAQ3['cbwd']=3
dataAQ4=dataAQ.loc[dataAQ['cbwd']=='SE']
dataAQ4['cbwd']=4
dataAQ5=dataAQ.loc[dataAQ['cbwd']=='cv']
dataAQ5['cbwd']=5

dataAQF=pd.concat([dataAQ1,dataAQ2,dataAQ3,dataAQ4,dataAQ5])


#цртање графикона
plt.figure()

plt.scatter(summer['No'],summer['PM_US Post'],c='orange',label='leto')
plt.scatter(spring['No'],spring['PM_US Post'],c='green',label= 'prolece')
plt.scatter(winter['No'],winter['PM_US Post'],c='blue',label='zima')
plt.scatter(autumn['No'],autumn['PM_US Post'],c='red',label='jesen')
plt.xlabel('2010              2011        2012         2013         2014       2015')
plt.ylabel('Koncentracija PM2.5 (ug/m**3)')
plt.legend(title="godisnja doba",loc='upper right')

plt.figure()

plt.scatter(ne['No'],ne['PM_US Post'],c='c', label='severoistocni')
plt.scatter(nw['No'],nw['PM_US Post'],c='b', label='severozapadni')
plt.xlabel('2010              2011        2012         2013         2014       2015')
plt.ylabel('Koncentracija PM2.5 (ug/m**3)')
plt.legend(title="vetrovi",loc='upper right')
plt.figure()
plt.scatter(sw['No'],sw['PM_US Post'],c='b', label='jugozapadni')
plt.scatter(se['No'],se['PM_US Post'],c='r', label='jugoistocni')
plt.xlabel('2010              2011        2012         2013         2014       2015')
plt.ylabel('Koncentracija PM2.5 (ug/m**3)')
plt.legend(title="vetrovi",loc='upper right')
plt.figure()
plt.scatter(cv['No'],cv['PM_US Post'],c='k', label='promenljivog_pravca')
plt.xlabel('2010              2011        2012         2013         2014       2015')
plt.ylabel('Koncentracija PM2.5 (ug/m**3)')
plt.legend(title="vetrovi",loc='upper right')


plt.figure()

plt.hist(dataAQ['AQ'],bins=15)
plt.plot(dataAQ['PM_US Post'].mean())
#norm=np.linalg.norm(dataAQ['PM_US Post'])

#plt.xlabel('2010              2011        2012         2013         2014       2015')
#plt.ylabel('Koncentracija PM2.5 (ug/m**3)')
plt.legend(title="БРОЈ САТИ СА ОДРЕЂЕНИМ ""\n"" КВАЛИТЕТОМ ВАЗДУХА ""\n"" HA-опасан ""\n"" VU-веома нездрав ""\n"" GO-добар ""\n"" MO-умерен ""\n"" UN-нездрав""\n"" US-нездрав за осетљиве групе",loc='upper left')

sns.displot(dataAQ, x="AQ")
sns.displot(dataAQF, x="HUMI", y="PM_US Post",ax=True)
sns.displot(dataAQF, x="AQ",y="PM_US Post")
sns.displot(dataAQF, x="Iws",y="PM_US Post")
sns.displot(dataAQF, x="season",y="PM_US Post")
sns.displot(dataAQF, x="DEWP",y="PM_US Post")
sns.displot(dataAQF, x="TEMP",y="PM_US Post")
sns.displot(dataAQF, x="cbwd",y="PM_US Post")
sns.displot(dataAQF, x="HUMI",y="AQ")

#укључивање интеракције између обележја
dataAQF['HUMIxAQ']=dataAQF['HUMI']*dataAQF['AQ']
dataAQF['cbwdxAQ']=dataAQF['cbwd']*dataAQF['AQ']
dataAQF['HUMIxcbwd']=dataAQF['HUMI']*dataAQF['cbwd']
dataAQF['DEWPxHUMI']=dataAQF['DEWP']*dataAQF['HUMI']
dataAQF['DEWPxAQ']=dataAQF['DEWP']*dataAQF['AQ']
dataAQF['AQxIws']=dataAQF['AQ']*dataAQF['Iws']

#додавањe базних функција 
dataAQF['e^AQ']=np.exp(dataAQF['AQ'])
dataAQF['(e^-Iws)+20']=-np.exp(-dataAQF['Iws'])+20
dataAQF['(e**sqrt(HUMI))/32']=np.exp(np.sqrt(dataAQF['HUMI']))/32


#интеракција између функција и других обележја
dataAQF['(e^AQ)xHUMI']=dataAQF['e^AQ']*dataAQF['HUMI']
dataAQF['DEWPxAQxHUMI']=dataAQF['DEWPxAQ']*dataAQF['HUMI']
dataAQF['(e^AQ)xAQ']=dataAQF['AQ']*dataAQF['e^AQ']



#матрица корелације
c=dataAQF.corr()
#sns.heatmap(c)


###################################################################################################


#примењујем линеарну регресију
#најпрецизнији модел добијам када не уклоним ниједно друго обележје

x=dataAQF.drop(columns=['PM_US Post','hour','month'])

y=dataAQF['PM_US Post']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=5)

first_regression_model = LinearRegression()

second_regression_model = Lasso()

third_regression_model = Ridge()

# Oбука
first_regression_model.fit(x_train, y_train)
second_regression_model.fit(x_train,y_train)
third_regression_model.fit(x_train,y_train)

# Tестирање
y1_predicted = first_regression_model.predict(x_test)
y2_predicted = second_regression_model.predict(x_test)
y3_predicted = third_regression_model.predict(x_test)

# Приказ
print("classic:")
print("////////////////////////////////////////////")
model_evaluation(y_test, y1_predicted, x_train.shape[0], x_train.shape[1])

print("////////////////////////////////////////////")
print("Lasso:")
print("////////////////////////////////////////////")
model_evaluation(y_test, y2_predicted, x_train.shape[0], x_train.shape[1])
print("////////////////////////////////////////////")
print("Ridge:")
model_evaluation(y_test, y3_predicted, x_train.shape[0], x_train.shape[1])
print("////////////////////////////////////////////")



































