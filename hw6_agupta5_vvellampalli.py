# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:25:04 2018

@author: Arpit, Deepak
"""

import numpy as np

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        
        # Initializing the weights randomly
        
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward (self,x, y):
        # Initializing gradients
        
        gradj_w = 0
        gradj_U = 0
        gradj_V = 0
        y_hat = []
        temp_U = np.zeros((1,6))
        temp_V = x[0]
        h = np.zeros((51,6))
        z = np.zeros((50,6))
        
        # running 51(len(x)+1) iterations to calculate the gradients
        for i in range(1,51):
            z[i-1,:] = np.dot(h[i-1,:],self.U) + np.dot(x[i-1],self.V).T[0]
            h[i,:] = np.tanh(z[i-1,:])
            y_hat.append(np.dot(self.w,h[i,:]))
            
            # Using the equation 1 from the derivation submitted in the pdf file
            gradj_w = gradj_w + np.dot((y_hat[i-1] - y[i-1]),h[i,:])
            
            temp_U = h[i-1,:]+np.dot(temp_U,np.dot(self.U,(np.diag(1-np.square(h[i-1,:])))))
            temp1 =  np.dot(y_hat[i-1]-y[i-1],self.w)
            temp2 = 1-np.square(h[i,:])
            temp3 = np.dot(temp1,temp2)
            
            # Using the equation 2 from the derivation submitted in the pdf file
            # This is a recursive term , so we used temp_U and kept updating it recursively
            gradj_U = gradj_U + np.dot(temp3,temp_U[0])

            temp_V = x[i-1] + np.dot(temp_V,np.dot(self.U,(np.diag(1-np.square(h[i-1,:])))))
            temp4 = np.dot(y_hat[i-1]-y[i-1],self.w)
            temp5 = 1-np.square(h[i,:])
            temp6 = np.dot(temp4,temp5)
            
            # Using the equation 3 from the derivation submitted in the pdf file
            # This is a recursive term , so we used temp_V and kept updating it recursively
            gradj_V = gradj_V + np.dot(temp6,temp_V[0])
            
        return gradj_w, gradj_U, gradj_V, h, z

    def forward (self,x,y,h,z):
        y_hat = []
        for i in range(1,51):
            y_hat.append(np.dot(h[i].T,self.w))
        cost = np.sum(0.5*np.square(y_hat - y))
        print(cost)        
        
    def SGD(self,x,y):
        # Using three different learning rates as V and U are recursive term
        # so they need to be learnt slower than the rate for W
        rate1 = 1e-1
        rate2 = 1e-5
        rate3 = 1e-7
        
        # Taking number of iterations as 100000
        for i in range(100000):
            gradj_w,gradj_U,gradj_V,h,z =self.backward(x,y)
            self.w = self.w - rate1*gradj_w
            self.U = self.U - rate2*gradj_U
            self.V = self.V - rate3*gradj_V
            self.forward(x,y,h,z)

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    #batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    print(xs)
    print(ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)

    # Calling the SGD function that in turns calls all the other functions required
    
    rnn.SGD(np.array(xs),np.array(ys))