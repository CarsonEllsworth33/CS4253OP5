# -*- coding: utf-8 -*-
import numpy as np

class image_grabber():
    
    def __init__(self,image_file,image_label):
        self.name1 = image_file
        self.name2 = image_label
        self.image_label_dict = {}
        self.image_file = open(image_file,"r")
        self.image_label = open(image_label,"r")
       
        
    
    def next_image(self):
        lst = self.image_file.readline().split(',')
        
        if not lst[0]:
            self.image_file.close()
            self.image_file = open(self.name1,"r")
            lst = self.image_file.readline().split(',')
        for x in range(len(lst)):
            lst[x] = int(lst[x])
        self.image_label_dict["image"] = np.array([lst])
        
        val = self.image_label.readline()
        if not val:
            self.image_label.close()
            self.image_label = open(self.name2,"r")
            val = int(self.image_label.readline() )
        self.image_label_dict["label"] = int(val)
        return self.image_label_dict