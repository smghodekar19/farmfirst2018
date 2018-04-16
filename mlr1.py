# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:22:46 2018

@author: NILESH
"""

# -*- coding: utf-8 -*-
import csv
import datetime
import os
import sys
def main():
    # print command line arguments
    area=sys.argv[1]
    return area
main()

class Farm:

    def conv(self,nutrient):
        nutrient_dict={'VL':1,'L':2,'M':3,'H':4,'VH':5 }
        return  nutrient_dict.__getitem__(nutrient)
    
    def temperature(self, area):
        try:
            with open('temprainfall.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                   if row[0] == area:
                       #print(row)
                       temperature=(float(row[2])+float(row[3]))/2
                       temp.append(round(temperature,2))
                       rainfall.append(float(row[4]))
        except IOError:
           print "No file exists named temperature.csv"   
           sys.exit("The required file does not exist!!!")               
        csvfile.close 
#print temp
#print rainfall
#print ()   
#csvfile.close           

##temp and rainfall code
#index=month_dict.__getitem__(month)
    def rainfall(self,prevtemp,prevrainfall):
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
    def nutrients(self):
        try:
            with open('nutrientsarea.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                   if row[0] == state:
                       narea=self.conv(row[1])
                       parea=self.conv(row[2])
                       karea=self.conv(row[3])
                       ph=row[4]
        except IOError:
           print "No file exists named nutrientsarea.csv"
           sys.exit("The required file does not exist!!!")               
        csvfile.close
#nutrient based filter
        try:
            
            with open('cropDB.csv', 'r') as csvfile, open('metacrops.csv', 'w') as metacrops:
                reader = csv.reader(csvfile)
                #writer=csv.writer(metacrops)
                metacrops.writelines("Crop, Rainfall, Temperature, pH \n")
                for row in reader:
                   ncrop=self.conv(row[8])
                   pcrop=self.conv(row[9])
                   kcrop=self.conv(row[10])
                   if(narea>=ncrop and parea>=pcrop and karea>=kcrop):
                       #swriter.writerows(row)
                       no_months=int(row[1])
                       if no_months<13:        ###todo for duration>12months
                       #if rainfall_final[no_months-1]>=float(row[6]) and rainfall_final[no_months-1]<=float(row[7]) and temp_final[no_months-1]>=float(row[2]) and temp_final[no_months-1]<=float(row[3]):
                           total=row[0]+","+str(rainfall_final[no_months-1])+","+str(temp_final[no_months-1])+","+ph+"\n"
                           metacrops.writelines(total)
                       #print total
        except IOError:
           print "No file exists named cropDB.csv",
           sys.exit("The required file does not exist!!!")     
        csvfile.close
        metacrops.close 

# rainfall and temp based filter
# rainfall temp ph

    def filewrite(self):
        n=1
        try:
            with open("metacrops.csv",'r') as f:
                with open("metacrops11.csv", "w") as f1:
                    for line in f:
                        if n==1:
                            n=n+1
                            continue
                        f1.write(line)
        except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
                sys.exit("No such file exists")
        f.close
        f1.close              
                    
#######################################

############MLR 

##########################################

# Data Preprocessing 

# Importing the libraries
    def regression(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        
        pred_crop = ""
        comma_flag=0
        n=0
        
        dataset=pd.read_csv('regressiondb.csv')
        locbased=pd.read_csv('metacrops.csv')
        
        try:
           with open('metacrops11.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                #metacrops.writelines("Crop,Production\n")
                #os.remove('final.txt')
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
                       if comma_flag==1:
                           pred_crop=pred_crop+", "
                       pred_crop= pred_crop + crop
                       comma_flag=1
                       #pred_crop=crop+","
                       #print (pred_crop),
                       
                #print(pred_crop)     
           csvfile.close
           return pred_crop
       
        except IOError:
            print "No file exists named metacrops11.csv"
            sys.exit("No such file exists")
        #os.remove('metacrops.csv')       
        #os.remove('metacrops11.csv')

if __name__ == '__main__':
    area=main()
    object=Farm()
    area=area+' '
    print(area)
    object.temperature(area)
    object.rainfall(prevtemp,prevrainfall)
    object.nutrients()
    object.filewrite()
    final_crop=object.regression()
    print (final_crop)
    reader = csv.DictReader(open('metasort.csv', 'r'))
    result = sorted(reader, key=lambda d: float(d['Production']),reverse=True)
    print(result)


area= 'Pune '
state='Maharashtra'
now = datetime.datetime.now()
month=now.month
month=3 #dummy month for july
temp=[]
temp_final=[]
rainfall=[]
rainfall_final=[]
prevtemp=0
prevrainfall=0
#pred_crop = ""

#Y_final=[][]
#month_dict = {'January':0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,'July': 6, 'August': 7, 'September': 8 , 'October': 9 , 'November': 10 , 'December': 11}

            
                       
'''reader = csv.DictReader(open('metasort.csv', 'r'))
        result = sorted(reader, key=lambda d: float(d['Production']),reverse=True)
        
        writer = csv.DictWriter(open('output.csv', 'w'), reader.fieldnames)
        writer.writeheader()
        writer.writerows(result)   
        
        import json
        jsonfile=open('sort.json','w')
        jsonfile.write(json.dumps(list(csv.reader(open('output.csv')))))
        
        os.remove('metacrops.csv')
        os.remove('metacrops11.csv')
'''

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

















