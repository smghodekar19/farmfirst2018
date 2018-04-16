# -*- coding: utf-8 -*-

import csv
import datetime


def conv(nutrient):
    nutrient_dict={'VL':1,'L':2,'M':3,'H':4,'VH':5 }
    return  nutrient_dict.__getitem__(nutrient)

area= 'Nasik '
state='Maharashtra'

now = datetime.datetime.now()
month=now.month
#month=7 dummy month for july

temp=[]
temp_final=[]
rainfall=[]
rainfall_final=[]
prevtemp=0
prevrainfall=0
Y_final=[]
#month_dict = {'January':0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,'July': 6, 'August': 7, 'September': 8 , 'October': 9 , 'November': 10 , 'December': 11}
with open('C:\Users\chandu\Desktop\server\code/temprainfall.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
       if row[0] == area:
           #print(row)
           temperature=(float(row[2])+float(row[3]))/2
           temp.append(round(temperature,2))
           rainfall.append(float(row[4]))
#print temp
#print rainfall
#print ()
csvfile.close           

##temp and rainfall code
#index=month_dict.__getitem__(month)
index=month-1
for i in range (1,13):
    #print prev
    prevtemp=prevtemp+temp[index]
    temp_final.append(round((prevtemp/i),2))
    prevrainfall=prevrainfall+rainfall[index]
    rainfall_final.append(round(prevrainfall,2))
    index= index+1
    if index==12:
        index=0
#print temp_avg
#print rainfall_final
        
        
# get nutrients of farmers area
with open('C:\Users\chandu\Desktop\server\code/nutrientsarea.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
       if row[0] == state:
           narea=conv(row[1])
           parea=conv(row[2])
           karea=conv(row[3])
           ph=row[4]
csvfile.close
#nutrient based filter
with open('C:\Users\chandu\Desktop\server\code/cropDB.csv', 'r') as csvfile, open('C:\Users\chandu\Desktop\server\code/metacrops.csv', 'w') as metacrops:
    reader = csv.reader(csvfile)
    #writer=csv.writer(metacrops)
    metacrops.writelines("Crop, Rainfall, Temperature, pH \n")
    for row in reader:
       ncrop=conv(row[8])
       pcrop=conv(row[9])
       kcrop=conv(row[10])
       if(narea>=ncrop and parea>=pcrop and karea>=kcrop):
           #swriter.writerows(row)
           no_months=int(row[1])
           if no_months<13:        ###todo for duration>12months
           #if rainfall_final[no_months-1]>=float(row[6]) and rainfall_final[no_months-1]<=float(row[7]) and temp_final[no_months-1]>=float(row[2]) and temp_final[no_months-1]<=float(row[3]):
               total=row[0]+","+str(rainfall_final[no_months-1])+","+str(temp_final[no_months-1])+","+ph+"\n"
               metacrops.writelines(total)
           #print total
           
csvfile.close
metacrops.close 

# rainfall and temp based filter
# rainfall temp ph
'''
#rather than filter keep those whose y_pred is >0 as done above
with open('metacrops.csv', 'r') as metacrops, open('metacrops1.csv', 'w') as metacrops1:
    reader = csv.reader(metacrops)
    metacrops1.writelines("Crop, Rainfall, Temperature, pH \n")
    for row in reader :
        no_months=int(row[1])
        if no_months<12:        ###todo for duration>12months
          # if rainfall_final[no_months-1]>=float(row[6]) and rainfall_final[no_months-1]<=float(row[7]) and temp_final[no_months-1]>=float(row[2]) and temp_final[no_months-1]<=float(row[3]):
          total=row[0]+","+str(rainfall_final[no_months-1])+","+str(temp_final[no_months-1])+","+ph+"\n"
          metacrops1.writelines(total)
                #print total
metacrops1.close
'''                
n=1
with open("C:\Users\chandu\Desktop\server\code/metacrops.csv") as f:
    with open("C:\Users\chandu\Desktop\server\code/metacrops11.csv", "w") as f1:
        for line in f:
            if n==1:
                n=n+1
                continue
            f1.write(line)
              
                
#######################################

############MLR 

##########################################

# Data Preprocessing 

# Importing the libraries
import sys
sys.path.append("numpy_path")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('C:\Users\chandu\Desktop\server\code/regressiondb.csv')
locbased=pd.read_csv('C:\Users\chandu\Desktop\server\code/metacrops.csv')
n=0
with open('C:\Users\chandu\Desktop\server\code/metacrops11.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
       crop=row[0]
# Importing the dataset
       metadata=dataset.loc[dataset['Crop'] == crop]
       X = metadata.iloc[:, :-2].values
       Y = metadata.iloc[:, 4].values
##fitting MLR to training set  
       regressor = LinearRegression()
       regressor.fit(X, Y)
#predecting test set results
#Y_pred = regressor.predict(X_test)
       X_locbased = locbased.loc[[n]].values
       X_locbased = X_locbased[:, 1:4]
       Y_pred=regressor.predict(X_locbased)
       if Y_pred>0:
           print "  ", crop,"  ",row, X_locbased, Y_pred
       n=n+1
       





# Encoding categorical data
# Encoding the Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

##dummy var trap removal
#X=X[:,1:]  ##no need to do this manually lib does it by default..written just for info.


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
##fitting MLR to training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#predecting test set results
Y_pred = regressor.predict(X_test)

Y_pred=regressor.predict(X_ip[0])
"""


#remove these comments as you complete one one of them
##delete metacrops and metacrops11 after use
#exception handling whereever posssible especially for files if not present
#does python have abstract static functions n all? make use of them too rather than traditional functions
#free space taken by variables etc manually to increase code size
# i am not getting logic for finding average of crops with duration more than 12 months plz help
# sorting data and writing it back to json file is left plz do if possible

















