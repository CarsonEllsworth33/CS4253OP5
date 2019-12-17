from node import node
import math
import numpy as np
from numpy import random as npr
import grab_image
from numba import jit,cuda



back_prop_per_image = 1

class neural_node(node):
    def __init__(self,func,name):
        super().__init__(name)
        self.act_func = func
        self.weights_out = None
        self.weights_in = None
        self.type = name[0]
        self.output = None

    def __repr(self):
        return self.name
        
    def set_activation_function(self,function):
        self.act_func = function
        
    def activate(self,x):
        if self.act_function == None:
            raise NotImplementedError
        else:
            val = self.act_func(x)
            #print(val)
            return val
            
    def generate_rand_weight(self):
        #for nnx in self._connections_out:
        #    self.weights.append(rd.random())
        self.weights_out = npr.rand(len(self.connections_out),1)
     
    
    def input_func(self):
        #possible problems with setup, might have to use dummy nodes so the input layer works
        return sum(self.weights_in[0][x] * ((self.connections_in[x].output) if self.connections_in and self.connections_in[x].output != None else 1) for x in range(len(self.weights_in)))
    
    def output_func(self):
        self.output = self.act_func(self.input_func())
        #print(self.name,self.output)
        return self.output
    
    @property 
    def set_weights_in(self):
        wi_arr = []
        for nnx in self.connections_in:
            wi_arr.append(nnx.weights_out[self.id_number][0])
        self.weights_in = np.array([wi_arr])
    
    @property
    def id_number(self):
        if (len(self.name) > 3 and self.type == 'h'):
            return int(self.name[3:])
        return int(self.name[2:]) if self.type == 'h' else int(self.name[1:])
        
        

def sigmoid(x):
    '''
    this is the sigmoid function, it is a smoothed out perceptron function
    '''
    return 1/(1 + math.exp(-x))

def sigmoidp(x):
    '''
    This is the derivative of the sigmoid function
    '''
    return sigmoid(x)*(1 - sigmoid(x))

def act1(x):
    return 0 if x < 0 else 1

def act2(x):
    return 1 if x < 0 else 0


class network():
    def __init__(self,layer,function,functionp,i,o,h=0):
        self.function = function
        self.functionp = functionp
        self.target_out_file = "labels.csv"
        self.hw_vector = None
        self.y_vector = None
        self.delta_j = []
        self.delta_k = []
        self.dummy_network = []
        if (layer >= 3):
            self.network = [[] for x in range(layer)]
            self.create_input_output_layer(i,o,function)
            self.create_hidden_layer(h,function)
            self.connect_nodes()
            
        elif(layer == 2):
            self.network = [[] for x in range(layer)]
            self.create_input_output_layer(i,o,function)
            self.connect_nodes()
            
        else:
            print("value not large enough, needs to be greater than or equal to 2")
            return
        
        for l in self.network:
            for nnx in l:
                nnx.generate_rand_weight()
                nnx.set_weights_in
                  
    def __repr__(self):
        return self.print_network
    
    
    def training(self,num_itter,ig):
        for x in range(num_itter):
            print("training instance {} of {}".format(x+1,num_itter))
            image_dat = ig.next_image()
            n1.train(image_dat["image"],outvect(image_dat["label"]))
    
    #@jit(target="cuda")
    def train(self,np_arr,y_vect):
        self.set_y_vect(y_vect)
        for nnx in self.network[0]:
            nnx.weights_in = np.array([[np_arr[0][nnx.id_number]]])
        for x in range(back_prop_per_image):
            self.hw_vector = self.weighted_input_sum_output()
            if((self.errvect==np.zeros(self.errvect.shape)).all()):
                break
            
            self.back_prop()
        
    def test(self,np_arr):
        for nnx in self.network[0]:
            nnx.weights_in = np.array([[np_arr[0][nnx.id_number]]])
        self.hw_vector = self.weighted_input_sum_output()
        return self.hw_vector[0]
        
    
    def create_input_output_layer(self,num_i,num_o,func):
        self.network[0] = [neural_node(func,"i{}".format(x)) for x in range(num_i)]
        self.network[-1] = [neural_node(func,"o{}".format(x)) for x in range(num_o)]

    def create_hidden_layer(self,num_h,func):
        l = len(self.network) - 1
        for x in range(1,l):
            self.network[x] = [neural_node(func,"h{}{}".format(x,y)) for y in range(num_h)]
        
    def connect_nodes(self):
        for i in range(len(self.network)):
            for nnx in self.network[i]:
                if i + 1 < len(self.network):
                    for nny in self.network[i+1]:
                        nnx.add_connection(nny)
        
    #@jit(target="cuda")                   
    def weighted_input_sum_output(self):
        vect = []
        for l in self.network:
            for nnx in l:
                if (l == self.network[-1]):
                    vect.append(nnx.output_func())
                else:
                    nnx.output_func()
        return np.array([vect])
        
                
    def input_setup(self,file):
        with open(file,"r") as input_data:
            for nnx in self.network[0]:
                nnx.weights_in = [float(input_data.readline())]
    
    #@jit(target="cuda")
    def hidden_weight_update(self,l,i,j):
        tmp = self.delt_j(l,i,j)
        self.network[l][i].weights_out[j] = self.network[l][i].weights_out[j] + self.alpha * self.network[l][i].output_func() * tmp
        return tmp
    
    #@jit(target="cuda")
    def output_weight_update(self,j,k):
        #wjk <- wjk + self.alpha * aj * delta_k
        tmp = self.delt_k(-2, k)
        self.network[-2][j].weights_out[k] = self.network[-2][j].weights_out[k] + self.alpha * self.network[-2][j].output_func() * tmp
        return tmp
    
    #@jit(target="cuda")
    def back_prop(self,l=None):
        #print("back_prop")
        if(l == None):
            l = (len(self.network)-2)
        if (l < 0):
            pass
        else:
            if(l == len(self.network)-2):
                #update weight rule for output nodes
                for nnx in self.network[l]:
                    for nny in nnx.connections_out: #these are the ouput nodes
                        #print("output Node:",nny.name)
                        nny.set_weights_in
                        self.delta_k.append(self.output_weight_update(nnx.id_number,nny.id_number))
                self.back_prop(l - 1)
            else:
                for nnx in self.network[l]:
                    for nny in nnx.connections_out:
                        nny.set_weights_in
                        self.delta_j.append(self.hidden_weight_update(l,nnx.id_number,nny.id_number))
                self.delta_k = self.delta_j.copy()
                self.delta_j.clear()
                self.back_prop(l - 1)        
    
    def delt_j(self,l,j,k):
        return self.functionp(self.network[l][j].input_func())*sum(self.network[l][j].weights_out[k] * self.delta_k[k])
                                                                   
    def delt_k(self,l,k):
        return self.errk(k) * self.functionp(self.network[l+1][k].input_func())
    
    def errk(self,k):
        vect = self.errvect
        return vect[0][k]
    
    def set_y_vect(self,label_arr):
        self.y_vector = np.array([label_arr])
    
    @property 
    def errvect(self):
        return np.array(self.y_vector - self.hw_vector)
    
    @property
    def alpha(self):
        return 1
    
    @property
    def print_network(self):
        #something weird happening here
        tmp = ""
        tmpllst = [] #layer list
        tmpstr = ""
        for l in self.network:
            if l is not self.network[-1]:
                for nnx in l:
                    tmpstr = ""
                    tmpstr += nnx.name + ":"
                    for nny in nnx.connections_out:
                        tmpstr += nny.name + ","
                    tmpllst.append(tmpstr)
                tmp+=str(tmpllst) + "\n"
        return tmp
    

def outvect(x):
    return [1 if x == y else 0 for y in range(10)]

ig = grab_image.image_grabber("training-first50.csv","testing-first50.csv")
ig.next_image()
n1 = network(4,sigmoid,sigmoidp,i=len(ig.image_label_dict["image"]),o=10,h=512)


     
        
