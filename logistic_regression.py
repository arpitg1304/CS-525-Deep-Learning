import numpy as np


trainingFaces = np.load('smile_data/trainingFaces.npy')
trainingLabels = np.load('smile_data/trainingLabels.npy')
testingFaces = np.load('smile_data/testingFaces.npy')
testingLabels = np.load('smile_data/testingLabels.npy')


def sigmoid(z):
    sigm = 1/(1+np.exp(-z))
    return sigm

def J(w,faces,labels,alpha = 0):
    yhat = sigmoid(np.dot(faces,w))
    temp = (alpha/2)*np.dot(w.T,w)
    first_term = np.dot(labels.T,np.log(yhat+1e-5))
    second_term = np.dot(1-labels.T,np.log(1-yhat+1e-5))
    cost = -(first_term+second_term)/2000 + temp
   #%cost = -1/2000 * np.sum(np.dot(labels, np.log(sigmoid(np.dot(faces,w))+1e-5))+np.dot(1-labels, np.log(1- sigmoid(np.dot(faces,w))+1e-5)))+(alpha/2)*np.dot(w.T,w)
    return cost

                                   
def gradJ (w, faces, labels, alpha = 0):
    grad_j = np.dot(faces.T,(sigmoid(np.dot(faces,w))-labels))+ alpha*w
    return grad_j


def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    rate = 1e-4
    tol = 1e-7
    cost_old = []
    cost_new = []
    while True:
        w_old = np.copy(w)
        w = w - rate*gradJ (w, trainingFaces, trainingLabels, alpha)
        cost_old = J(w_old, trainingFaces, trainingLabels, alpha)
        cost_new  = J(w, trainingFaces, trainingLabels, alpha)
        print(cost_new)
        #print(cost_new)
        if (np.absolute(cost_old - cost_new) < tol):
            break
    return w

def method4 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e-3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels,alpha)

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print((J(w, trainingFaces, trainingLabels, alpha)))
    print((J(w, testingFaces, testingLabels, alpha)))



w4 = method4(trainingFaces, trainingLabels, testingFaces, testingLabels)

print('Costs for Training and Testing Faces')
reportCosts(w4, trainingFaces, trainingLabels, testingFaces, testingLabels)
