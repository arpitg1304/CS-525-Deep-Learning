# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:03:55 2018

@author: Arpit; Deepak
"""
import time
start_time = time.time()
import numpy as np
#mport matplotlib.pyplot as plt

def reshape(X):
    X =  np.reshape(X, (X.shape[0], 28,28))
    return X

#Loading datasets

train_faces = (np.load('mnist_train_images.npy'))
train_faces_reshaped = reshape(np.load('mnist_train_images.npy'))
train_labels = (np.load('mnist_train_labels.npy'))
test_faces = (np.load('mnist_test_images.npy'))
test_faces_reshaped = reshape(np.load('mnist_test_images.npy'))
test_labels = (np.load('mnist_test_labels.npy'))


def softmax(z):
    sm = (np.exp(z) / np.sum(np.exp(z),axis=1)[: , np.newaxis])  
    return sm

def y_hat(faces,w):
    z = np.dot(faces, w)
    prob = (np.exp(z) / np.sum(np.exp(z),axis=1)[: , np.newaxis])
    return prob
    
def J(w,faces,labels,alpha = 0):
    #yhat = y_hat(faces,w)
    m = train_faces.shape[0]
    #temp1 = np.dot(yhat, np.log(yhat))
    prob = y_hat(faces,w)
    loss = (-1 / m) * np.sum(labels * np.log(prob+1e-5))
    #+ (1e-3/2)*np.sum(np.dot(w.T,w))
    return loss
                                   
def gradJ (w, faces, labels, alpha = 0):
    prob = y_hat(faces, w)
    grad = np.dot(train_faces.T,(prob - labels))
    #+ 1e-3*w
    return grad

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.zeros([train_faces.shape[1], 10])
    rate = 1e-5
    cost_new = []
    for i in range(0,500):
        print("Iteration: "+str(i))
        #w_old = np.copy(w)
        w = w - (rate*gradJ(w, trainingFaces, trainingLabels, alpha))
        #cost_old = J(w_old, trainingFaces, trainingLabels, alpha)
        if(i>479):
            cost_new  = J(w, trainingFaces, trainingLabels, alpha)
            print("The cost value is: "+str(cost_new))
        #print(cost_new)
        
        
    return w

def soft_regression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e-2
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels,alpha)

#Comment next two lines after running the program once
W = soft_regression(train_faces, train_labels, test_faces, test_labels)
np.save('trained2', W)

#Uncomment the next line to run the code with the saved weights
#W = np.load('trained2.npy')

predictions = np.dot(train_faces, W)
p1 = np.argmax(predictions,axis=1)
p2 = np.argmax(train_labels,axis=1)

accuracy1 = sum(p1 == p2)/(float(len(p2)))

print("Accuracy on Training Data: " + str(accuracy1*100)+ " percent")

predictions_test = np.dot(test_faces, W)
p3 = np.argmax(predictions_test,axis=1)
p4 = np.argmax(test_labels,axis=1)
accuracy2 = sum(p3 == p4)/(float(len(p4)))

print("Accuracy on Test Data: " + str(accuracy2*100)+ " percent")
print("--- %s seconds ---" % (time.time() - start_time))
