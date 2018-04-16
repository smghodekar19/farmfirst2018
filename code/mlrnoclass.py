# -*- coding: utf-8 -*-
"""
"""

# -*- coding: utf-8 -*-
import csv
import datetime
import os
import sys

def main():
    #print command line arguments
    area=sys.argv[1]
    area=area+' '
   # print(area)
    return str(area)

area=main()
#area= 'Pune '

#state='Maharashtra'
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

def conv(nutrient):
    nutrient_dict={'VL':1,'L':2,'M':3,'H':4,'VH':5 }
    return  nutrient_dict.__getitem__(nutrient)

def temperature():
    with open('code/temprainfall.csv') as csvfile:
        #print('this is area:'+area)
        reader = csv.reader(csvfile)
        flag=0
        for row in reader:
            #print("this is area:"+area)
            if row[0] == area:
               #print(row)
               if flag==0:
                   state=row[1]
                   flag=1
               temperature=(float(row[3])+float(row[4]))/2
               temp.append(round(temperature,2))
               rainfall.append(float(row[5]))
               
    return state           
    csvfile.close 
    
state=temperature()   
#print temp
#print rainfall
#print ()   
#csvfile.close           

##temp and rainfall code
#index=month_dict.__getitem__(month)
#def rainfall(prevtemp,prevrainfall):
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
def nutrients(state):
    try:
        with open('code/nutrientsarea.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
               if row[0] == state:
                   narea=conv(row[1])
                   parea=conv(row[2])
                   karea=conv(row[3])
                   ph=row[4]
    except IOError:
       print "No file exists named nutrientsarea.csv"
       sys.exit("The required file does not exist!!!")               
    csvfile.close
#nutrient based filter
    try:
        
        with open('code/cropDB.csv', 'r') as csvfile, open('code/metacrops.csv', 'w') as metacrops:
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
    except IOError:
       print "No file exists named cropDB.csv",
       sys.exit("The required file does not exist!!!")     
    csvfile.close
    metacrops.close 

# rainfall and temp based filter
# rainfall temp ph

def filewrite():
    n=1
    try:
        with open("code/metacrops.csv",'r') as f:
            with open("code/metacrops11.csv", "w") as f1:
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

def regression():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    
    n=0
    crop_Y_pred=[]
    crop_name=[]
    dataset=pd.read_csv('code/regressiondb.csv')
    locbased=pd.read_csv('code/metacrops.csv')
    
    try:
       with open('code/metacrops11.csv', 'r') as csvfile:
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
                   crop_Y_pred.append(round(Y_pred[0],3))
                   crop_name.append(crop)
                                     
                   '''if comma_flag==1:
                       pred_crop=pred_crop+","
                       #crop_final_dict[crop].append(Y_pred[0])
                   pred_crop= pred_crop + crop
                   comma_flag=1'''
       #print(crop_name,crop_Y_pred) 
       sorted_crops=quicksort(crop_name,crop_Y_pred,0,len(crop_Y_pred)-1)                       
       csvfile.close
       #return pred_crop
       return sorted_crops
   
    except IOError:
        print "No file exists named metacrops11.csv"
        sys.exit("No such file exists")
    #os.remove('metacrops.csv')       
    #os.remove('metacrops11.csv')
        
                   
def quicksort(crop_name,crop_Y_pred,start, end):
    if start < end:
        # partition the list
        pivot = partition(crop_name,crop_Y_pred, start, end)
        # sort both halves
        quicksort(crop_name,crop_Y_pred, start, pivot-1)
        quicksort(crop_name,crop_Y_pred, pivot+1, end)
    return crop_name

def partition(crop_name,crop_Y_pred, start, end):
    pivot = crop_Y_pred[start]
    left = start+1
    right = end
    done = False
    while not done:
        while left <= right and crop_Y_pred[left] >= pivot:
            left = left + 1
        while crop_Y_pred[right] <= pivot and right >=left:
            right = right -1
        if right < left:
            done= True
        else:
            # swap places Y_pred
            temp=crop_Y_pred[left]
            crop_Y_pred[left]=crop_Y_pred[right]
            crop_Y_pred[right]=temp
            
            # swap places Y_crop
            temp1=crop_name[left]
            crop_name[left]=crop_name[right]
            crop_name[right]=temp1
            
    # swap start with myList[right]
    temp=crop_Y_pred[start]
    crop_Y_pred[start]=crop_Y_pred[right]
    crop_Y_pred[right]=temp
    
    # swap start with myList[right] for crop 
    temp1=crop_name[start]
    crop_name[start]=crop_name[right]
    crop_name[right]=temp1
        
    return right    
    
    
    
    
    
    
    
    
    
    

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

def ListtoStr(sorted_crop):
    pred_crop = ""
    comma_flag=0
    for i in range (len(sorted_crop)):
        if comma_flag==1:
            pred_crop=pred_crop+","
                       
        pred_crop= pred_crop + sorted_crop[i]
        comma_flag=1
    
    return pred_crop
    

if __name__ == '__main__':
    
    #object=Farm()
    #print(area)
    #temperature()
    #rainfall(prevtemp,prevrainfall)
    nutrients(state)
    filewrite()
    sorted_crop=regression()
    final_crop=ListtoStr(sorted_crop)
    print (final_crop)
    
    
'''    object.temperature(area)
    object.rainfall(prevtemp,prevrainfall)
    object.nutrients()
    object.filewrite()
    final_crop=object.regression()
'''    
    
'''reader = csv.DictReader(open('metasort.csv', 'r'))
result = sorted(reader, key=lambda d: float(d['Production']),reverse=True)
print(result)'''

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
"""



'''
changes done: 
    state added in temprainfall file
    state variable updated from temprainfall file
    tempfile is now updated in all locations!
    
    sorting function divided into 2 parts
    QS used
    list to str function added
    final string obtained as required
    done!

'''













