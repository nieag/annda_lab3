import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import animation as animation


def loadData(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        data = [line.split(',') for line in lines]
        data = np.asarray([int(i) for i in data[0]])

    return data.reshape(-1,32*32)

def sign(x):
    #np.sign returns 0 for x==0....
    return (x<0)*(-1) + (x>=0)

def generateData():
    return [[-1, -1, 1, -1, 1, -1, -1, 1], [-1, -1, -1, -1, -1, 1, -1, -1,], [-1, 1, 1, -1, -1, 1, -1, 1]]

def generateNoisyData():
    return [[1, -1, 1, -1, 1, -1, -1, 1], [1, 1, -1, -1, -1, 1, -1, -1], [1, 1, 1, -1, 1, 1, -1, 1]]

def compW(X, N=1024, zeroDiag=None):
    Id = 1-np.eye(len(X[0]))
    W = np.zeros((len(X[0]),len(X[0])))
    for x in X:
        W += np.outer(x,x)
    return W

def recallAll(X,W):
    out = []
    for x in X:
        out.append(recall(x,W))
    return list(out)

def recall(x,W):
    return list(sign(np.dot(W,x)))

def findAttractors(W):
    size = W.shape[0]
    keys = [-1,1]
    X = [",".join(map(str, comb)) for comb in product(keys, repeat=size)]

    temp = []
    for x in X:
        line = x.split(",")
        temp.append([int(i) for i in line])

    rec = np.copy(temp)
    for i in range(15):
        rec = recallAll(rec,W)
    unique_data = [list(x) for x in set(tuple(x) for x in rec)]
    return unique_data

def recallAsync(x, W, iter=5):

    steps=np.arange(W.shape[0])
    steps=[int(z) for z in steps]

    x_temp = np.copy(x)
    saved_ims = []
    count = 0
    E = []

    for n in range(iter):
        choices = np.copy(steps)
        for i in steps:
            node = np.random.choice(choices, replace=False)
            x_temp[node] = sign(np.dot(x_temp, W[:,node]))
            if count == 100:
                saved_ims.append(np.copy(x_temp))
                count = 0
            count+=1
            E.append(energy(x_temp, W))
    saved_ims.append(x_temp)

    return saved_ims,E

def plotIms(X, exciteIm, recalledIm, save=0, append="", path="test"):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
    fig, ax = plt.subplots(2,3)
    print(len(recalledIm))
    with writer.saving(fig, path, len(exciteIm)):
        for i in range(len(recalledIm)):
            plt.subplot(2,3,1)
            plt.imshow(X[0].reshape(32,32))
            plt.title("Train img1")
            plt.subplot(2,3,2)
            plt.imshow(X[1].reshape(32,32))
            plt.title("Train img2")
            plt.subplot(2,3,3)
            plt.imshow(X[2].reshape(32,32))
            plt.title("Train img3")

            plt.subplot(2,3,4)
            plt.imshow(exciteIm.reshape(32,32))
            plt.title("Excite img")

            plt.subplot(2,3,5)
            plt.imshow(recalledIm[i].reshape(32,32))
            plt.title("Recalled image")

            fig.suptitle("Number of itertions: {}".format(i*100))

            writer.grab_frame()
            print(i)

    plt.show()

def energy(x,W):
    return -np.dot(np.dot(x,W),x.T)

def addNoise(img, noiseLevel):
    mask = np.random.choice([-1, 1], size=(len(img),), p=[noiseLevel, 1-noiseLevel])
    return np.multiply(img,mask)

def compW2(X, N,P):
    ro = (1/(N*P))*np.sum(X)
    W = np.zeros((len(X[0]),len(X[0])))
    Id = 1-np.eye(len(X[0]))

    for x in X:
        W += np.outer(x-ro,x-ro)
    return W

def recall2(X, W, bias):
    x = 0.5 + 0.5*sign((W@X) -bias)
    return x

#------------------------------------------------------------
def assignment3_0():
    X = generateData()
    W = compW(X)
    print(recall(X[0],W)) #exactly the same as X[0]

def assignment3_1():
    X = generateData()
    X_noisy = generateNoisyData()

    print("X: {}".format(X))
    W = compW(X)
    #print(recallAll(X_noisy,W))

    print("Number of attractors: {}".format(findAttractors(W)))

def assignment3_2():
    X = loadData('pict.dat') #loads 32*32x11
    W = compW(X[:3,:],iter=10)

    exciteIm = X[10]

    recalledIms, = recallAsync(exciteIm, W)

    plotIms(X, exciteIm, recalledIms, save=1, append="", path="")

    #recalledIm = np.array(recall(exciteIm, W))

def assignment3_3():
    X = loadData('pict.dat') #loads 32*32x11
    attractor = X[:3]
    W = compW(attractor, N=1024)

    exciteIm = X[10]
    dummy,E = recallAsync(exciteIm, W)

    fig = plt.figure()
    plt.plot(E)
    plt.title("Energy plot of image 11")
    plt.xlabel("Training steps")
    plt.ylabel("Energy")
    #fig.savefig("Energy_image_11.png")

    X = [np.zeros(X[0].shape), np.zeros(X[0].shape), np.zeros(X[0].shape)]
    W = np.random.randn(1024,1024)
    recalledIms,E = recallAsync(exciteIm, W)
    fig = plt.figure()
    plt.plot(E)
    plt.title("Energy plot of image 11 with random weights")
    plt.xlabel("Training steps")
    plt.ylabel("Energy")
    #fig.savefig("Energy_image_11_random_weights.png")
    plotIms(X, exciteIm, recalledIms, save=1, append="", path="random_weights.mp4")

    W = np.random.randn(1024,1024)
    W = (W + W.T)/2
    recalledIms,E = recallAsync(exciteIm, W)
    fig = plt.figure()
    plt.plot(E)
    plt.title("Energy plot of image 11 with random, symmetric weights")
    plt.xlabel("Training steps")
    plt.ylabel("Energy")
    #fig.savefig("Energy_image_11_random_weights_symmetric.png")

    #plotIms(X, exciteIm, recalledIms, save=1, append="", path="random_weights_symmetric.mp4")

def assignment3_4():
    #add noise to images and try to recall
    #how much noise can be added?
    X = loadData('pict.dat') #loads 32*32x11
    attractor = X[:3]

    X_noise = []
    W = compW(attractor, iter=1, N=1024)
    noiseStep = 0.1
    noOfImages = 3
    labels = []
    noOfIterations = 5

    for j in range(noOfImages):
        for i in range(int(1/noiseStep)):
            X_noise.append(addNoise(X[j],noiseStep*(i+1)))
            labels.append(j)

    for i in range(noOfIterations):
        X_noise = recallAll(X_noise,W)

    recalledPattern = []
    for i in range(len(X_noise)):
        if (np.sum(X[0] - X_noise[i]) <= 0):
            recalledPattern.append(0)
        elif (np.sum(X[1] - X_noise[i]) <= 0):
            recalledPattern.append(1)
        elif (np.sum(X[2] - X_noise[i]) <= 0):
            recalledPattern.append(2)
        else:
            recalledPattern.append(-1)
    print(recalledPattern)

    for i in range(3):
        fig = plt.figure()
        plt.imshow(np.array(X[i]).reshape(32,32))
        fig.savefig("img{}.png".format(i))

    N = int(1/noiseStep)
    for i in range(noOfImages):
        fig, ax = plt.subplots(2,5)

        for n in range(N):
            plt.subplot(2,5,n+1)
            plt.imshow(np.array(X_noise[i*N + n]).reshape(32,32))
            plt.title("noise: {}".format(round((n+1)*(noiseStep),2)))

        fig.savefig("recalled_{}_noisy".format(i))

def assignment3_5():
    X = loadData('pict.dat') #loads 32*32x11

    performance = []

    for i in range(1,8):
        W = compW(X[:i])
        noOfRecalled = 0
        for n in range(i+1):
            x_temp = np.copy(X[n])
            x_noise = addNoise(x_temp, 0.3)
            x_temp = recall(x_noise,W)
            if (np.sum(x_temp-X[n]) == 0):
                noOfRecalled+=1
        performance.append(noOfRecalled)
    print(performance)

def assignment3_5_2():
    noOfUnits = 100
    noOfPatterns = 300
    X_rand = []
    performance = []
    bias = 0.5
    noise = 0.1

    for i in range(noOfPatterns):
        mask = np.random.choice([1, -1], size=(noOfUnits,), p=[bias,1-bias])
        X_rand.append(mask)

    for i in range(1,len(X_rand)+1):
        W = compW(X_rand[:i], zeroDiag=1)
        noOfRecalled = 0
        for n in range(0,i):
            x_temp = np.copy(X_rand[n])
            x_prev = np.copy(x_temp)
            x_temp = addNoise(x_temp,noise)
            x_temp = recall(x_temp, W)
            if np.all(x_temp == x_prev):
                noOfRecalled+=1
        performance.append(noOfRecalled)

    print(performance)
    performance_norm = np.divide(performance, np.arange(noOfPatterns)+1)
    #print(performance_norm)

    catForgettingLimit = noOfUnits*0.138

    fig, ax = plt.subplots(2,1, figsize=(8, 9))

    plt.subplot(2,1,1)
    plt.plot(performance)
    plt.axvline(x=catForgettingLimit, c='r')
    plt.title("Number of stable patterns stored")
    plt.ylabel("# of stable patterns recalled")
    plt.xlabel("# of patterns learned")

    plt.subplot(2,1,2)
    plt.plot(performance_norm)
    plt.axvline(x=catForgettingLimit, c='r')
    plt.title("Number of stable patterns stored normalised on patters learned")
    plt.ylabel("Fraction of stable patterns recalled")
    plt.xlabel("# of patterns learned")
    fig.savefig("zero_diag_0_1_noise.png")

def assignment3_5_2_extra():
    noOfUnits = 100
    noOfPatterns = 300

    iterations = 1
    total_performance = []
    total_performance_norm = []

    for m in range(iterations):
        print(m)
        performance = []
        X_rand = []

        for i in range(noOfPatterns):
            mask = np.random.choice([1, -1], size=(noOfUnits,))
            X_rand.append(mask)

        for i in range(1,len(X_rand)+1):
            W = compW(X_rand[:i])
            #print(X_rand[:i])
            noOfRecalled = 0
            for n in range(0,i):
                #print(n)
                x_temp = np.copy(X_rand[n])
                x_temp = recall(x_temp,W)
                if (np.sum(x_temp-X_rand[n]) == 0):
                    noOfRecalled+=1
            performance.append(noOfRecalled)
        total_performance.append(performance)

    total_performance = np.sum(total_performance,0)/iterations
    total_performance_norm = np.divide(total_performance,np.arange(noOfPatterns)+1)
    catForgettingLimit = noOfUnits*0.18

    fig, ax = plt.subplots(2,1, figsize=(8, 9))
    fig.suptitle("Mean of {} runs".format(iterations))

    plt.subplot(2,1,1)
    plt.plot(total_performance)
    plt.axvline(x=catForgettingLimit, c='r')
    plt.title("Number of patterns stored")
    plt.ylabel("# of patterns recalled")
    plt.xlabel("# of patterns learned")

    plt.subplot(2,1,2)
    plt.plot(total_performance_norm)
    plt.axvline(x=catForgettingLimit, c='r')
    plt.title("Number of patterns stored normalised on patters learned")
    plt.ylabel("Fraction of patterns recalled")
    plt.xlabel("# of patterns learned")
    fig.savefig("recalled_patterns_mean.png")

def assignment3_6():

    X = []
    noOfPatterns = 300
    noOfUnits = 100
    bias = np.arange(0.001, 20, 0.1)
    performance = []
    activity = 0.1
    best = []

    for i in range(noOfPatterns):
        x = np.random.choice([1, 0], size=(noOfUnits,), p=[activity, 1-activity])
        X.append(x)

    for b in bias:
        print(b)
        for i in range(1,len(X)+1):
            W = compW2(X[:i], noOfUnits, i)
            noOfRecalled = 0
            for n in range(0,i):
                x_temp = np.copy(X[n])
                x_prev = np.copy(x_temp)
                #x_temp = addNoise(x_temp,0.05)
                for k in range(10):
                    x_temp = recall2(x_temp, W, b)
                    #print(x_temp)
                    if np.sum(x_temp-x_prev) == 0:
                        noOfRecalled+=1
                        break

            performance.append(noOfRecalled)
        best.append(np.max(performance))

    fig= plt.figure(figsize=(8, 9))
    plt.plot(bias, best)

    fig.savefig("test.png")

if __name__ == '__main__':
    #assignment3_0()
    #assignment3_1()
    #assignment3_2()
    #assignment3_3()
    #assignment3_4()
    #assignment3_5()
    assignment3_5_2()
    #assignment3_5_2_extra()
    #assignment3_6()
