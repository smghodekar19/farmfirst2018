#!/usr/bin/Python
import json
import sys  
import os


filepath = 'C:\Users\chandu\Desktop\server\file\user0.json'  
with open(filepath ) as fp:  
   line = fp.readline()
 
   while line:
       print(" {}".format( line.strip()))
       line = fp.readline()
     
 