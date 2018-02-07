import numpy as np

trainingFaces = np.load('smile_data/trainingFaces.npy')
trainingLabels = np.load('smile_data/trainingLabels.npy')
testingFaces = np.load('smile_data/testingFaces.npy')
testingLabels = np.load('smile_data/testingLabels.npy')

def whiten(faces):
    face = faces.T
    apl = 1e-3
    temp1_old = np.dot(face,face.T)
    I = np.eye(576)
    temp1 = temp1_old + apl * I
    eigval,eigvec = np.linalg.eig(temp1)
    #eigval[0:24] = eigval[0:24] + 1e-11
    eigval_temp = 1./np.sqrt(eigval)
    eigval_new = np.diag(eigval_temp)
    temp2 = np.dot(eigvec.T,face)
    transform = np.dot(eigval_new,temp2)      
    transformed = transform.T
    #print np.shape(transformed)
    return np.real(transformed), eigval_new, eigvec

def J (w, faces, labels, alpha = 0.):
    cost = np.sum ( 0.5 * (faces.dot(w)-labels)**2 ) + (alpha/2)*np.dot(w.T,w)
    return cost


def gradJ (w, faces, labels, alpha = 0.):
    grad_j = np.dot(faces.T,(np.dot(faces,w)-labels))+ alpha*w
    return grad_j


def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    rate = 0.25
    tol = 1e-3
    cost_old = []
    cost_new = []
    while True:
        w_old = np.copy(w)
        w = w - rate*gradJ (w, trainingFaces, trainingLabels, alpha)
        cost_old = J(w_old, trainingFaces, trainingLabels, alpha)
        cost_new  = J(w, trainingFaces, trainingLabels, alpha)
        print(cost_new)
        if (np.absolute(cost_old - cost_new) < tol):
            break
    return w

def method4 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 0
    trainingFaces_whiten, _, _ = whiten(trainingFaces)
    return gradientDescent(trainingFaces_whiten, trainingLabels, testingFaces, testingLabels, alpha)


def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    trainingFaces_whiten, _, _ = whiten(trainingFaces)
    print((J(w, trainingFaces_whiten, trainingLabels, alpha)))
    #testingFaces_whiten = whiten(testingFaces)
    a, b, c = whiten(trainingFaces)
    temp2 = np.dot(c.T,testingFaces.T)
    transform = np.dot(b,temp2)      
    transformed = transform.T
    testingFaces_whiten = np.real(transformed)
    
    print((J(w, testingFaces_whiten, testingLabels, alpha=1e-3)))

print('Trajectory of Gradient Descent for the whitened training faces\n')
w4 = method4(trainingFaces, trainingLabels, testingFaces, testingLabels)

print('Costs of Training and Testing\n')

reportCosts(w4, trainingFaces, trainingLabels, testingFaces, testingLabels)
