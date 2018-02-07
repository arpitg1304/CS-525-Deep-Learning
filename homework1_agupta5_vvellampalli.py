
# coding: utf-8

# Author : Arpit Gupta , Deepak Vellampalli ; Deep Learning HW 1
# Student ID: 860227466

# In[149]:

import numpy as np


# In[150]:

def problem1(A,B):
    print("Answer of Problem 1\n")
    return print(A + B)


# In[151]:

def problem2(A,B,C):
    print('\n')
    print(("Answer of Problem 2\n"))
    return print(np.dot(A,B) - C) 
    


# In[152]:

def problem3(A,B,C):
    print('\n')
    print(("Answer of Problem 3\n"))
    return print(A * B + C.T) 


# In[153]:

def problem4(X,Y):
    print('\n')
    print(("Answer of Problem 4\n"))
    return print(np.dot(X.T , Y))


# In[154]:

def problem5(A):
    print('\n')
    print(("Answer of Problem 5\n"))
    x,y = A.shape
    return print(np.zeros([x,y]))    


# In[155]:

def problem6(A):
    print('\n')
    print(("Answer of Problem 6\n"))
    x,y = A.shape
    return print(np.ones([x]))    


# In[156]:

def problem7(A):
    print('\n')
    print(("Answer of Problem 7\n"))
    return print(np.linalg.inv(A))


# In[157]:

def problem8(A,x):
    print('\n')
    print(("Answer of Problem 8\n"))
    x.shape = A.shape[0],1
    return print(np.linalg.solve(A,x))


# In[158]:

def problem9(A, x):
    print('\n')
    print(("Answer of Problem 9\n"))
    x.shape = 1,A.shape[0]
    return print((np.linalg.solve(A.T,x.T)).T)


# In[159]:

def problem10(A, alpha):
    print('\n')
    print(("Answer of Problem 10\n"))
    x,y = A.shape
    return print(A + alpha * (np.eye(x,y)))


# In[160]:

def problem11(A,i,j):
    print('\n')
    print(("Answer of Problem 11\n"))
    return print(A[i-1,j-1])


# In[161]:

def problem12(A,i):
    print('\n')
    print(("Answer of Problem 12\n"))
    return print(np.sum(A[i-1, :]))


# In[162]:

def problem13(A,c,d):
    print('\n')
    print(("Answer of Problem 13\n"))
    return print(np.mean(A[np.where(np.logical_and(A>=c,A<=d))]))


# In[169]:

def problem14(A,k):
    print('\n')
    print(("Answer of Problem 14\n"))
    p,q = np.linalg.eig(A)
    sort_id = np.argsort(p)[-k:] #sorting and selecting k number of elements
    return print(q[:, sort_id[::-1]]) #reversing


# In[170]:

def problem15(x,k,m,s):
    print('\n')
    print(("Answer of Problem 15\n"))
    n = x.size
    x.shape = n,1
    z = np.ones([n,1])
    mu = x + m * z
    sig = np.sqrt(s*np.eye(n,n))
    return print(np.dot(sig,np.random.randn(n,k))+mu)  


# In[175]:

A = np.matrix('1 5 2; 2 3 1; 3 1 2')
B = np.matrix('1 1 1; 1 0 1; 2 0 1')
C = np.matrix('1 0 1; 2 1 1; 1 1 1')
x = np.array([2,2,2])
y = np.array([2,3,4])
problem1(A,B)
problem2(A,B,C)
problem3(A,B,C)
problem4(x,y)
problem5(A)
problem6(A)
problem7(A)
problem8(A,x)
problem9(A,x)
problem10(A,5)
problem11(A,1,2)
problem12(A,3)
problem13(A,1,2)
problem14(A,2)
problem15(x,3, 6, 7)

