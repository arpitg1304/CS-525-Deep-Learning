import numpy as  np
import math

trainingFaces = np.load('mnist_train_images.npy')
trainingLabels = np.load('mnist_train_labels.npy')
testingFaces = np.load('mnist_test_images.npy')
testingLabels = np.load('mnist_test_labels.npy')
validationFaces = np.load('mnist_validation_images.npy')
validationLabels = np.load('mnist_validation_labels.npy')

#trainingFaces shape - 55000*784
#trainingLabels shape - 55000*10
#testingFaces shape - 10000*784
#testingLabels shape - 10000*10

def relu(z1):
    rel = np.maximum(np.zeros(np.shape(z1)),z1)
    return rel

def relu_p(z1): 
    z1[z1 >0] = 1
    z1[z1<=0] = 0        
    return z1

def softmax(z2):
    temp1 = np.exp(z2) #for z>700 set z=700 if z<-700 z=-700
    temp2 = np.sum(temp1,axis=1)[:,None]
    softm = temp1/temp2
    return softm

def findBestHyperparameters(a):
    hl_units = np.array([30,40,50])
    l_rate = np.array([0.001,0.005,0.01,0.05,0.1,0.5])
    mini_b = np.array([16,32,64,128,256])
    epoch_n = np.array([80,100,125,150])

    hlu = np.random.choice(hl_units)
    lr = np.random.choice(l_rate)
    mb = np.random.choice(mini_b)
    ep = np.random.choice(epoch_n)
    print('choosen hyperparameters')
    print(hlu)
    print(lr)
    print(mb)
    print(ep)
    return hlu,lr,mb,ep

def J(w1,w2,b1,b2,faces,labels,m,alpha1,alpha2):
    z1 = np.dot(faces,w1)+b1
    h1 = relu(z1)
    z2 = np.dot(h1,w2)+b2
    yhat = softmax(z2)
    temp3 = np.multiply(labels,np.log(yhat+1e-5)) 
    cost = np.sum(temp3) + alpha1* (np.linalg.norm(w1))+alpha2*(np.linalg.norm(w2))
    return -cost/m

def gradJ (w1,w2,b1,b2,faces, labels,alpha1,alpha2,hlu,lr,mb,ep):
    z1 = np.dot(faces,w1)+b1
     
    h1 = relu(z1)
    z2 = np.dot(h1,w2)+b2
    yhat = softmax(z2)
    
    g = -np.divide(labels,yhat)
    h2 = yhat

    g = yhat - labels
    gradj_b2 = np.dot(np.ones((1,mb)),g)
    gradj_w2 = np.dot(h1.T,g)+alpha2*w2
    g = np.dot(g,w2.T)
    g = g * relu_p(z1)
    gradj_b1 = np.dot(np.ones((1,mb)),g)
    gradj_w1 = np.dot(faces.T,g)+alpha1*w1   
    return gradj_b1,gradj_b2,gradj_w1,gradj_w2


def SGD(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha1,alpha2,hlu,lr,mb,ep):
    w1 = np.random.randn(trainingFaces.shape[1],hlu)
    b1 = np.random.randn(1,hlu)
    w2 = np.random.randn(hlu,trainingLabels.shape[1])
    b2 = np.random.randn(1,10)

    rate = 1e-3
    tol = 1e-4
    cost_old = []
    cost_new = []
    epoch = 1
    while epoch< ep:
        w1_old = np.copy(w1)
        w2_old = np.copy(w2)
        b1_old = np.copy(b1)
        b2_old = np.copy(b2)

        x = math.floor(55000/mb)
        #print(x)
        
        for i in range(0,int(x*mb),mb):
            trainingFaces_b = trainingFaces[i:i+mb,:]
            trainingLabels_b = trainingLabels[i:i+mb,:]            
            gradj_b1,gradj_b2,gradj_w1,gradj_w2 = gradJ (w1,w2,b1,b2,trainingFaces_b, trainingLabels_b, alpha1,alpha2,hlu,lr,mb,ep)        
            w1 = w1 - rate*gradj_w1
            
            w2 = w2 - rate*gradj_w2
            
            b1 = b1 - rate*gradj_b1
            
            b2 = b2 - rate*gradj_b2
            
        epoch = epoch+1
        print("Epoch number: " +str(epoch))
        cost_old = J(w1_old, w2_old, b1_old, b2_old, trainingFaces, trainingLabels,55000, alpha1, alpha2)
        cost_new  = J(w1, w2, b1, b2, trainingFaces, trainingLabels,55000, alpha1, alpha2)
        print(cost_new)
        #if (np.absolute(cost_old - cost_new) < tol):
            #break
    return w1,w2,b1,b2

def reportCosts (w1,w2,b1,b2, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha1,alpha2):
    print("The cost for train data is")
    print((J(w1,w2,b1,b2, trainingFaces, trainingLabels,55000, 0, 0)))
    print("The cost for validation data is")
    validation_cost = J(w1,w2,b1,b2, validationFaces, validationLabels,5000, 0, 0)
    print(validation_cost)
    print("The cost for test data is")
    print((J(w1,w2, b1,b2, testingFaces, testingLabels,10000, 0, 0)))
    return validation_cost
    


def method5 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 0
    hlu,lr,mb,ep = findBestHyperparameters(0)
#    hlu = 50
#    lr = 0.5
#    mb = 32
#    ep = 125
    w1,w2,b1,b2 = SGD(trainingFaces, trainingLabels, testingFaces, testingLabels,0.01,0.01,hlu,lr,mb,ep)
    return w1,w2,b1,b2, hlu,lr,mb,ep

def method_best_params (trainingFaces, trainingLabels, testingFaces, testingLabels, hlu_best, lr_best, mb_best, ep_best):
    alpha = 0
    w1_best, w2_best, b1_best, b2_best = SGD(trainingFaces, trainingLabels, testingFaces, testingLabels,0.01,0.01,hlu_best,lr_best,mb_best,ep_best)
    return w1_best,w2_best,b1_best,b2_best

for i in range(1,11):
    print('Round')
    print(i)
    w1,w2,b1,b2, hlu,lr,mb,ep = method5(trainingFaces, trainingLabels, testingFaces, testingLabels)
    validation_cost = reportCosts(w1,w2,b1,b2, trainingFaces, trainingLabels, testingFaces, testingLabels,0.01,0.01)
    if i ==1:
        ss = np.array([validation_cost,  hlu,lr,mb,ep])
    else:
        if ss[0] > validation_cost:
            ss = np.array([validation_cost, hlu,lr,mb,ep])
            
print('The best set of hyperparameters and the corresponding cost for validation set is')
print(ss)


            
w1,w2,b1,b2, hlu,lr,mb,ep = method5(trainingFaces, trainingLabels, testingFaces, testingLabels)
reportCosts(w1,w2,b1,b2, trainingFaces, trainingLabels, testingFaces, testingLabels,0,0)

hlu_best = ss[0]#50
lr_best = ss[1]#0.5
mb_best = ss[2]#32
ep_best = ss[3]#125

w1_best,w2_best,b1_best,b2_best = method_best_params(trainingFaces, trainingLabels, testingFaces, testingLabels, hlu_best, lr_best, mb_best, ep_best )
reportCosts(w1_best,w2_best,b1_best,b2_best, trainingFaces, trainingLabels, testingFaces, testingLabels,0,0)


def get_prediction(w1, w2, b1, b2, faces):
    z1 = np.dot(faces,w1)+b1
     
    h1 = relu(z1)
    z2 = np.dot(h1,w2)+b2
    yhat = softmax(z2)
    return yhat

def get_accuracy(predictions, labels):
    p1 = np.argmax(predictions,axis=1)
    p2 = np.argmax(labels,axis=1)
    accuracy = sum(p1 == p2)/(float(len(p2)))
    return accuracy
    

predictions_train = get_prediction(w1, w2, b1, b2, trainingFaces);
predictions_test = get_prediction(w1, w2, b1, b2, testingFaces);

accuracy_train = get_accuracy(predictions_train, trainingLabels)
accuracy_test = get_accuracy(predictions_test, testingLabels)


print('The accuracy for Training data is: ' + str(accuracy_train))
print('The accuracy for Testing data is: ' + str(accuracy_test))
















    

    

    

    



