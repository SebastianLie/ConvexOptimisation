import numpy as np
import math
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt

### DSA3102 CONVEX OPTIMISATION HW1  ###
### Author: Sebastian Lie ###

mat = loadmat('HW1data.mat')

# by default 
testx = mat["Xtest"] 
trainx = mat["Xtrain"]  # (3065,57)
testy = mat["ytest"]  # (3065,1)
trainy = mat["ytrain"]

# Accuracy dictionary made using grid search
acc_dict = {'Armijo beta = 0.1, sigma = 0.1, initial = 0': (0.9290364583333334, 195.51127672195435, 1414),
            'Armijo beta = 0.1, sigma = 0.2, initial = 0': (0.9290364583333334, 196.40087056159973, 1414),
            'Armijo beta = 0.1, sigma = 0.3, initial = 0': (0.9290364583333334, 195.7096071243286, 1414),
            'Armijo beta = 0.1, sigma = 0.4, initial = 0': (0.9290364583333334, 195.77836728096008, 1414),
            'Armijo beta = 0.2, sigma = 0.1, initial = 0': (0.9290364583333334, 121.56685972213745, 883),
            'Armijo beta = 0.2, sigma = 0.2, initial = 0': (0.9290364583333334, 121.77729296684265, 883),
            'Armijo beta = 0.2, sigma = 0.3, initial = 0': (0.9290364583333334, 121.65076088905334, 883),
            'Armijo beta = 0.2, sigma = 0.4, initial = 0': (0.9290364583333334, 121.62672233581543, 883),
            'Armijo beta = 0.3, sigma = 0.1, initial = 0': (0.9290364583333334, 266.47731757164, 1935),
            'Armijo beta = 0.3, sigma = 0.2, initial = 0': (0.9290364583333334, 266.22399044036865, 1937),
            'Armijo beta = 0.3, sigma = 0.3, initial = 0': (0.9290364583333334, 265.9267325401306, 1940),
            'Armijo beta = 0.3, sigma = 0.4, initial = 0': (0.9290364583333334, 269.9959063529968, 1940),
            'Armijo beta = 0.4, sigma = 0.1, initial = 0': (0.9290364583333334, 118.86411881446838, 863),
            'Armijo beta = 0.4, sigma = 0.2, initial = 0': (0.9290364583333334, 118.42331576347351, 863),
            'Armijo beta = 0.4, sigma = 0.3, initial = 0': (0.9290364583333334, 118.39830565452576, 863),
            'Armijo beta = 0.4, sigma = 0.4, initial = 0': (0.9290364583333334, 118.93395256996155, 863),
            'Armijo beta = 0.5, sigma = 0.1, initial = 0': (0.9290364583333334, 99.81205224990845, 724),
            'Armijo beta = 0.5, sigma = 0.2, initial = 0': (0.9290364583333334, 99.76034188270569, 724),
            'Armijo beta = 0.5, sigma = 0.3, initial = 0': (0.9290364583333334, 99.97069764137268, 724),
            'Armijo beta = 0.5, sigma = 0.4, initial = 0': (0.9290364583333334, 199.71015739440918, 1448),
            'Armijo beta = 0.6, sigma = 0.1, initial = 0': (0.9290364583333334, 89.46670746803284, 649),
            'Armijo beta = 0.6, sigma = 0.2, initial = 0': (0.9290364583333334, 89.38792634010315, 649),
            'Armijo beta = 0.6, sigma = 0.3, initial = 0': (0.9290364583333334, 148.95760369300842, 1082),
            'Armijo beta = 0.6, sigma = 0.4, initial = 0': (0.9290364583333334, 149.07728338241577, 1082),
            'Armijo beta = 0.7, sigma = 0.1, initial = 0': (0.9290364583333334, 84.01133441925049, 607),
            'Armijo beta = 0.7, sigma = 0.2, initial = 0': (0.9290364583333334, 120.2623519897461, 867),
            'Armijo beta = 0.7, sigma = 0.3, initial = 0': (0.9290364583333334, 119.86942911148071, 868),
            'Armijo beta = 0.7, sigma = 0.4, initial = 0': (0.9290364583333334, 120.01704430580139, 868),
            'Armijo beta = 0.8, sigma = 0.1, initial = 0': (0.9290364583333334, 101.90744209289551, 730),
            'Armijo beta = 0.8, sigma = 0.2, initial = 0': (0.9290364583333334, 101.70398688316345, 730),
            'Armijo beta = 0.8, sigma = 0.3, initial = 0': (0.9290364583333334, 101.40980052947998, 731),
            'Armijo beta = 0.8, sigma = 0.4, initial = 0': (0.9290364583333334, 126.6762318611145, 914),
            'Armijo beta = 0.9, sigma = 0.1, initial = 0': (0.9290364583333334, 89.11970686912537, 637),
            'Armijo beta = 0.9, sigma = 0.2, initial = 0': (0.9290364583333334, 89.3410484790802, 637),
            'Armijo beta = 0.9, sigma = 0.3, initial = 0': (0.9290364583333334, 99.19865036010742, 708),
            'Armijo beta = 0.9, sigma = 0.4, initial = 0': (0.9290364583333334, 110.02369856834412, 787),
            'Armijo beta = 0.1, sigma = 0.1, initial = 1': (0.9361979166666666, 34.913686752319336, 254),
            'Armijo beta = 0.1, sigma = 0.2, initial = 1': (0.9348958333333334, 30.813579320907593, 223),
            'Armijo beta = 0.1, sigma = 0.3, initial = 1': (0.9348958333333334, 30.897355318069458, 223),
            'Armijo beta = 0.1, sigma = 0.4, initial = 1': (0.9348958333333334, 30.549256324768066, 223),
            'Armijo beta = 0.2, sigma = 0.1, initial = 1': (0.9348958333333334, 40.04691982269287, 292),
            'Armijo beta = 0.2, sigma = 0.2, initial = 1': (0.9348958333333334, 37.74006009101868, 275),
            'Armijo beta = 0.2, sigma = 0.3, initial = 1': (0.9348958333333334, 37.774964809417725, 275),
            'Armijo beta = 0.2, sigma = 0.4, initial = 1': (0.9348958333333334, 37.694212913513184, 275),
            'Armijo beta = 0.3, sigma = 0.1, initial = 1': (0.935546875, 44.94977164268494, 324),
            'Armijo beta = 0.3, sigma = 0.2, initial = 1': (0.935546875, 43.878644704818726, 317),
            'Armijo beta = 0.3, sigma = 0.3, initial = 1': (0.935546875, 44.396249771118164, 317),
            'Armijo beta = 0.3, sigma = 0.4, initial = 1': (0.9348958333333334, 37.20848369598389, 271),
            'Armijo beta = 0.4, sigma = 0.1, initial = 1': (0.9348958333333334, 29.413305044174194, 214),
            'Armijo beta = 0.4, sigma = 0.2, initial = 1': (0.9348958333333334, 29.47915530204773, 214),
            'Armijo beta = 0.4, sigma = 0.3, initial = 1': (0.9348958333333334, 29.70651149749756, 215),
            'Armijo beta = 0.4, sigma = 0.4, initial = 1': (0.9348958333333334, 29.619826555252075, 215),
            'Armijo beta = 0.5, sigma = 0.1, initial = 1': (0.9348958333333334, 31.935614585876465, 232),
            'Armijo beta = 0.5, sigma = 0.2, initial = 1': (0.9348958333333334, 32.078205585479736, 233),
            'Armijo beta = 0.5, sigma = 0.3, initial = 1': (0.9348958333333334, 32.36146783828735, 234),
            'Armijo beta = 0.5, sigma = 0.4, initial = 1': (0.9348958333333334, 39.08145785331726, 281),
            'Armijo beta = 0.6, sigma = 0.1, initial = 1': (0.9361979166666666, 28.10730767250061, 193),
            'Armijo beta = 0.6, sigma = 0.2, initial = 1': (0.935546875, 34.708186626434326, 235),
            'Armijo beta = 0.6, sigma = 0.3, initial = 1': (0.935546875, 31.874711513519287, 230),
            'Armijo beta = 0.6, sigma = 0.4, initial = 1': (0.935546875, 31.88076162338257, 230),
            'Armijo beta = 0.7, sigma = 0.1, initial = 1': (0.9368489583333334, 24.866475105285645, 178),
            'Armijo beta = 0.7, sigma = 0.2, initial = 1': (0.9368489583333334, 24.9862003326416, 178),
            'Armijo beta = 0.7, sigma = 0.3, initial = 1': (0.9361979166666666, 24.76472806930542, 177),
            'Armijo beta = 0.7, sigma = 0.4, initial = 1': (0.935546875, 23.955928325653076, 172),
            'Armijo beta = 0.8, sigma = 0.1, initial = 1': (0.935546875, 24.680019855499268, 175),
            'Armijo beta = 0.8, sigma = 0.2, initial = 1': (0.935546875, 25.08188533782959, 177),
            'Armijo beta = 0.8, sigma = 0.3, initial = 1': (0.935546875, 24.94025468826294, 176),
            'Armijo beta = 0.8, sigma = 0.4, initial = 1': (0.9361979166666666, 29.952916860580444, 212),
            'Armijo beta = 0.9, sigma = 0.1, initial = 1': (0.935546875, 25.394136667251587, 175),
            'Armijo beta = 0.9, sigma = 0.2, initial = 1': (0.935546875, 25.42099952697754, 175),
            'Armijo beta = 0.9, sigma = 0.3, initial = 1': (0.935546875, 25.49980592727661, 176),
            'Armijo beta = 0.9, sigma = 0.4, initial = 1': (0.935546875, 28.44990372657776, 197),
            'Armijo beta = 0.1, sigma = 0.1, initial = -1': (0.9296875, 152.00957250595093, 1101),
            'Armijo beta = 0.1, sigma = 0.2, initial = -1': (0.9296875, 151.39109420776367, 1101),
            'Armijo beta = 0.1, sigma = 0.3, initial = -1': (0.9296875, 152.66269373893738, 1110),
            'Armijo beta = 0.1, sigma = 0.4, initial = -1': (0.9296875, 159.33481121063232, 1164),
            'Armijo beta = 0.2, sigma = 0.1, initial = -1': (0.9296875, 91.6049964427948, 666),
            'Armijo beta = 0.2, sigma = 0.2, initial = -1': (0.9296875, 91.62103128433228, 666),
            'Armijo beta = 0.2, sigma = 0.3, initial = -1': (0.9296875, 91.52620959281921, 666),
            'Armijo beta = 0.2, sigma = 0.4, initial = -1': (0.9296875, 91.57408046722412, 666),
            'Armijo beta = 0.3, sigma = 0.1, initial = -1': (0.9309895833333334, 73.21619772911072, 531),
            'Armijo beta = 0.3, sigma = 0.2, initial = -1': (0.9296875, 61.06863021850586, 445),
            'Armijo beta = 0.3, sigma = 0.3, initial = -1': (0.9296875, 61.27112674713135, 445),
            'Armijo beta = 0.3, sigma = 0.4, initial = -1': (0.9296875, 61.77374505996704, 449),
            'Armijo beta = 0.4, sigma = 0.1, initial = -1': (0.9309895833333334, 37.69517493247986, 274),
            'Armijo beta = 0.4, sigma = 0.2, initial = -1': (0.9309895833333334, 37.940553426742554, 275),
            'Armijo beta = 0.4, sigma = 0.3, initial = -1': (0.9309895833333334, 38.10611057281494, 275),
            'Armijo beta = 0.4, sigma = 0.4, initial = -1': (0.9309895833333334, 38.23473882675171, 278),
            'Armijo beta = 0.5, sigma = 0.1, initial = -1': (0.9309895833333334, 39.05155420303345, 283),
            'Armijo beta = 0.5, sigma = 0.2, initial = -1': (0.9309895833333334, 38.90597224235535, 283),
            'Armijo beta = 0.5, sigma = 0.3, initial = -1': (0.9309895833333334, 39.05856370925903, 284),
            'Armijo beta = 0.5, sigma = 0.4, initial = -1': (0.9309895833333334, 40.121748208999634, 292),
            'Armijo beta = 0.6, sigma = 0.1, initial = -1': (0.9303385416666666, 28.228463649749756, 204),
            'Armijo beta = 0.6, sigma = 0.2, initial = -1': (0.9322916666666666, 26.34553861618042, 190),
            'Armijo beta = 0.6, sigma = 0.3, initial = -1': (0.9303385416666666, 42.08839464187622, 303),
            'Armijo beta = 0.6, sigma = 0.4, initial = -1': (0.9303385416666666, 42.59607529640198, 303),
            'Armijo beta = 0.7, sigma = 0.1, initial = -1': (0.9309895833333334, 38.732409715652466, 278),
            'Armijo beta = 0.7, sigma = 0.2, initial = -1': (0.9303385416666666, 36.570191860198975, 263),
            'Armijo beta = 0.7, sigma = 0.3, initial = -1': (0.9322916666666666, 32.94886827468872, 236),
            'Armijo beta = 0.7, sigma = 0.4, initial = -1': (0.9303385416666666, 32.43821382522583, 233),
            'Armijo beta = 0.8, sigma = 0.1, initial = -1': (0.9309895833333334, 31.97647714614868, 227),
            'Armijo beta = 0.8, sigma = 0.2, initial = -1': (0.9309895833333334, 34.62938165664673, 246),
            'Armijo beta = 0.8, sigma = 0.3, initial = -1': (0.9322916666666666, 33.516358852386475, 237),
            'Armijo beta = 0.8, sigma = 0.4, initial = -1': (0.9303385416666666, 33.26203012466431, 236),
            'Armijo beta = 0.9, sigma = 0.1, initial = -1': (0.9309895833333334, 30.66697907447815, 213),
            'Armijo beta = 0.9, sigma = 0.2, initial = -1': (0.9309895833333334, 33.178274393081665, 231),
            'Armijo beta = 0.9, sigma = 0.3, initial = -1': (0.9309895833333334, 32.55791258811951, 226),
            'Armijo beta = 0.9, sigma = 0.4, initial = -1': (0.9322916666666666, 34.49678921699524, 240)}


## line search methods  ##

def gprime(w,d,t,X,y):
    res = 0
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(w,X[i])) * np.exp(-y[i] * t * np.dot(d,X[i]))
        numerator = -y[i]*np.dot(np.transpose(d),X[i])*exp_part
        res = res +(numerator/(1+exp_part))
    print(res)
    return res[0]

def gprimeprime(w,d,t,X,y):
    res = 0 
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i])) * np.exp(-y[i] *t* np.dot(np.transpose(d),X[i]))
        numerator = (-y[i]*np.dot(np.transpose(d),X[i])**2)*exp_part
        res = res +(numerator/(1+exp_part)**2)
    print(res[0])
    return res[0]
    
def newtons(w, d, tol):
    t = 1
    while abs(gprime(w,d,t,trainx,trainy)) > tol:
        
        t = t - (gprime(w,d,t,trainx,trainy)/gprimeprime(w,d,t,trainx,trainy))
    return t

def golden_search(w, d, a, b, maxit, tol):

    phi = (sqrt(5.0) - 1)/2.0
    lam = b - phi*(b - a)
    mu = a + phi *(b-a)
    flam = loglikelihood(w+lam*d,trainx,trainy)
    fmu = loglikelihood(w+mu*d,trainx,trainy)
    for i in range(maxit):
        if flam > fmu:
            a = lam
            lam = mu
            mu = a + phi*(b-a)
            fmu = loglikelihood(w+mu*d,trainx,trainy)
        else:
            b = mu
            mu = lam
            fmu = flam
            lam = b - phi*(b-a)
            flam = loglikelihood(w+lam*d,trainx,trainy)

        if (b-a) <= tol:
            break
    return (b-a)/2

def bisection(a,b,tol):

    maxit = 10000
    la = loglikelihood(a, trainx, trainy)
    lb = loglikelihood(b, trainx, trainy)
    
    for i in range(maxit):
        x = (a+b)/2
        lx = loglikelihood(x, trainx,trainy)
        if (lx * lb <= 0):
            a = x
            la = lx
        else:
            b = x
            lb = lx
        if (b-a) < tol:
            break
    return x


def armijo(alpha_bar,w,d,beta,sigma):
    fx0 = loglikelihood(w,trainx,trainy)
    alpha = alpha_bar
    delta = np.dot(loglikelihood_grad(w,trainx,trainy),d)
    while loglikelihood(w+alpha*d,trainx,trainy) >  fx0 + alpha*sigma*delta:
        alpha = beta * alpha
    return alpha

######################################

## Objective function and its gradient ##

def loglikelihood(w, X, y):
    result = 0
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i]))
        result += np.log(1+exp_part)
    return result

def loglikelihood_grad(w, X, y):
    grad = np.zeros(57)
    for i in range(len(X)):
        exp_part = np.exp(-y[i] * np.dot(np.transpose(w),X[i]))
        numerator = -y[i]*exp_part*X[i]
        grad = grad +(numerator/(1+exp_part))
    return grad

######################################

## Steepest descent ##


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def steepest_descent(X,y,w0,maxit,tol,*line_search_params):  # use varargs
    step_size = 1
    w = w0  # inital guess
    line_search_method = line_search_params[0]
    obj_value_list = list()
    num_iter = 0
    for i in range(maxit):
        
        grad = loglikelihood_grad(w,trainx,trainy)
        d = -grad
        norm_d = np.linalg.norm(d)
        if norm_d < tol:
            break
        obj_value = loglikelihood(w,trainx,trainy)[0]
        obj_value_list.append(obj_value)
        print("Iteration {0}: obj = {1:9.3f}".format(i,obj_value))
        if norm_d < tol:
            break
        else:
            
            if line_search_method == "armijo":
                beta = line_search_params[1]
                sigma = line_search_params[2]
                w_prev = w
                step_size = armijo(step_size,w,d,beta,sigma)
                
            elif line_search_method == "fixed":
                step_size = line_search_params[1]
                w_prev = w
                
            elif line_search_method == "exact":
                step_size = newtons(w,d,0.1)

            elif line_search_method == "diminishing":
                step_size = line_search_params[1]/math.sqrt(i)
                
            else:  # default is armijos
                w_prev = w
                step_size = armijo(step_size,w,d,0.7,0.2)

            # update 
            w = w +  step_size * d
            num_iter += 1
            
    return w, obj_value_list, num_iter 

######################################

## Helper functions that produce useful things ##

def predict(w, xtest, ytest):
    if len(w) != len(xtest):
        w = np.transpose(w)
    yhat = sigmoid(np.dot(xtest,w))
    yhat = np.fromiter(map(lambda x: 1 if x > 0.5 else -1,yhat),dtype=np.double)
    correct = 0
    for i in range(len(yhat)):
	    if yhat[i] == ytest[i]:
		    correct += 1
    return correct/len(yhat)


# Vimpt
def test_grad(): # proof that grad function works well enough
    alp = 1*10**(-8)
    x = np.random.rand(57)
    differences = list()
    for i in range(57):
        e0 = np.zeros(57)
        e0[i] = 1 # test 0th part
        diff = loglikelihood_grad(x,trainx,trainy)[i] - (loglikelihood(x+alp*e0,trainx,trainy)-loglikelihood(x, trainx,trainy))/alp
        differences.append(round(diff[0],5))
    return differences
'''
Produced:
[-0.01087, -0.00607, -0.00211, -0.00653, -0.00775, -0.00695, -0.01511, -0.00564, -0.01248, -0.00927, -0.01077, -0.00991, -0.00842, -0.00685, -0.00771, -0.007,
-0.01319, -0.01054, -0.0055, -0.00613, -0.01242, -0.00824, -0.00657, -0.00809, -0.00528, -0.00606, -0.0072, -0.00767, -0.00747, -0.0079, -0.00731, -0.00784,
-0.00672, -0.00775, -0.00796, -0.01021, -0.00604, -0.00764, -0.00567, -0.00743, -0.00753,
-0.00705, -0.00747, -0.00642, -0.00621, -0.00564, -0.00703, -0.00686, -0.005, -0.00376, -0.00468, 0.00207, -0.00642, -0.00411, -0.00089, -0.00049, -0.00891]
'''


def GridSearch():
    
    # find best armijo parameters and initial solution
    for b in np.arange(0.1,1,0.1): # use arange cos need to iterate through floats
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,np.zeros(57),100000,0.5,"armijo",b,s)
            end = time.time()
            acc_dict["w0 = 0, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)
    for b in np.arange(0.1,1,0.1):
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,np.ones(57),100000,0.5,"armijo",b,s)
            end =time.time()
            acc_dict["w0 = 1, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)
    for b in np.arange(0.1,1,0.1):
        for s in np.arange(0.1,0.5,0.1):
            start =time.time()
            wA = steepest_descent(trainx,trainy,-1*np.ones(57),100000,0.5,"armijo",b,s)
            end = time.time()
            acc_dict["w0 = -1, Armijo beta = {0}, sigma = {1}".format(b,s)] = (predict(wA,testx,testy),end-start)


def plot_results():
    
    wA, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.2)
    wA2, armijo_obj_values2, num_iter2 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.1)

    fig, ax = plt.subplots()
    ax.plot(armijo_obj_values, range(1,num_iter1+1), 'r',label="beta = 0.7, sigma = 0.2") 
    ax.plot(armijo_obj_values2, range(1,num_iter2+1), 'b',label="beta = 0.7, sigma = 0.1") 
    plt.xlabel("Objective function values, Armijo's Step Size Strategy")
    plt.ylabel("Number of iterations")
    legend = ax.legend(loc='upper right',fontsize='small')
    plt.show()

    wf, fixed_obj_values, num_iter3 = steepest_descent(trainx,trainy,np.ones(57),100000,150,"fixed",0.001)
    wf2, fixed_obj_values2, num_iter4 = steepest_descent(trainx,trainy,np.ones(57),100000,150,"fixed",0.0005)

    fig, ax = plt.subplots()
    ax.plot(fixed_obj_values, range(1,num_iter3+1), 'r',label="Step Size = 0.001")  
    ax.plot(fixed_obj_values2, range(1,num_iter4+1), 'b',label="Step Size = 0.0005") 
    plt.xlabel("Objective function values, Fixed Step Size Strategy")
    plt.ylabel("Number of iterations")
    legend = ax.legend(loc='upper right',fontsize='small')
    plt.show()

#GridSearch()
#plot_results()
start = time.time()
wA1, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,2*np.ones(57),100000,10,"armijo",0.7,0.1)
end = time.time()
#wA2, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.7,0.4)

#wA3, armijo_obj_values, num_iter1 = steepest_descent(trainx,trainy,np.ones(57),100000,100,"armijo",0.6,0.1)
print(predict(wA1,trainx,trainy))
print(end-start)
print(predict(wA1,testx,testy))
#print(predict(wA2,trainx,trainy))
#print(predict(wA3,trainx,trainy))

