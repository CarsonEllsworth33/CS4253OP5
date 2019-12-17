# -*- coding: utf-8 -*-

import time
import numpy as np


def split_nkmdb_data(file):
    arr = []
    cnt = 0
    for line in open(file,"r"):
        if cnt == 0:
            cnt+=1
        else:
            lnarr = line.split(",")
            print(lnarr)
            #reason for this split is to convert the string to a list, if 
            # list(lnarr[1]) was used it would weirdly seperate every 
            # to its own element
            lnmd = lnarr[1].split() + lnarr[3:6] + lnarr[10:12] + lnarr[12:17]
            arr.append(lnmd)
            cnt+=1
            
    return arr
    
def filter_nkmdb_data(file):
    arr = split_nkmdb_data(file)
    print(len(arr))
        
    for launch in arr:
        if launch[-1].lower() == "success":
            launch[-1] = 1    
        elif launch[-1].lower() == "failure":
            launch[-1] = 0
        elif launch[-1].lower() == "unknown":
            i = arr.index(launch)
            arr.pop(i)
            
    new_data = open("modified_data.csv","w")
    for line in arr:
        for elem in line:
            if(elem == line[-1]):
                new_data.write("{}".format(elem))
                break
            new_data.write("{},".format(elem))
        new_data.write("\n")
        
    print(len(arr)) 
    print(arr)
    