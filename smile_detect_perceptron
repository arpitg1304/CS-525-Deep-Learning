
# coding: utf-8

# # Authors: Arpit Gupta, Deepak Vellampalli ; Deep Learning HW 2

# In[265]:

import numpy as np


# In[285]:

def J (w, faces, labels, alpha = 0.):
    #print(labels)
    #wt = w.T
    cost = np.sum ( 0.5 * (faces.dot(w)-labels)**2 ) + (alpha/2)*np.dot(w.T,w)
    #cost = j /2
    #((alpha/2)*
    return cost


# # Method 1
# # Setting gradient 0

# In[286]:

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    w = np.linalg.solve(np.dot(trainingFaces.T, trainingFaces), np.dot(trainingFaces.T, trainingLabels))
    #cost_train = J(w, trainingFaces, trainingLabels, 0)
    #print(cost_train)
    #cost_test = J(w, testingFaces, testingLabels, 0)
    #print("dvbdejkvbfevl fenje fwn")
    #print(cost_test)
    return w


# In[287]:


# # Method 2 : Gradient Descent

# In[288]:

def gradJ (w, faces, labels, alpha = 0.):
    grad_j = np.dot(faces.T,(np.dot(faces,w)-labels))+ alpha*w
    return grad_j


# In[289]:

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    rate = 7e-6
    tol = 1e-3
    cost_old = []
    cost_new = []
    while True:
        w_old = np.copy(w)
        w = w - rate*gradJ (w, trainingFaces, trainingLabels, alpha)
        cost_old = J(w_old, trainingFaces, trainingLabels, alpha)
        cost_new  = J(w, trainingFaces, trainingLabels, alpha)
        #print(cost_old - cost_new)
        if (np.absolute(cost_old - cost_new) < tol):
            break
    return w


# In[290]:

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    w = gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)
    return w


# In[291]:

def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)


# In[292]:

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print((J(w, trainingFaces, trainingLabels, alpha)))
    print((J(w, testingFaces, testingLabels, alpha)))


# In[293]:

trainingFaces = np.load('smile_data/trainingFaces.npy')
trainingLabels = np.load('smile_data/trainingLabels.npy')
testingFaces = np.load('smile_data/testingFaces.npy')
testingLabels = np.load('smile_data/testingLabels.npy')


w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
#print(w1)
w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)

w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)

for w in [ w1, w2, w3]:
    reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)

a = np.linalg.norm(w2)**2
b = np.linalg.norm(w3)**2

print ("Norm of W2 is ",a)
print ("Norm of W3 is ",b)

print("Norm of W2 is much higher than W3")

#print ("\nThe value of |w|^2 in Method 2: {}".format(np.linalg.norm(w2)**2))
#print ("The value of |w|^2 in Method 3: {}".format(np.linalg.norm(w3)**2))
