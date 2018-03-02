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

def compW(X, iter=1):
    Id = np.eye(len(X[0]))
    W = np.zeros((len(X[0]),len(X[0])))
    for i in range(iter):
        for x in X:
            W += np.outer(x,x)
    return W-Id

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

    rec = recallAll(temp,W)
    unique_data = [list(x) for x in set(tuple(x) for x in rec)]
    return unique_data

def recallAsync(x, W, iter=5):

    steps=np.arange(W.shape[0])
    steps=[int(z) for z in steps]

    x_temp = np.copy(x)
    saved_ims = []
    count = 0
    for n in range(iter):
        choices = np.copy(steps)
        for i in steps:
            node = np.random.choice(choices, replace=False)
            x_temp[node] = sign(np.dot(x_temp, W[:,node]))
            if count == 100:
                saved_ims.append(np.copy(x_temp))
                count = 0
            count+=1
    saved_ims.append(x_temp)

    return saved_ims

def plotIms(X, exciteIm, recalledIm, save=0, append="", path=""):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
    ims = []
    fig, ax = plt.subplots(2,3)
    with writer.saving(fig, "writer_test.mp4", len(exciteIm)):
        for i in range(len(exciteIm)):

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

    plt.show()



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

    print("Number of attractors: {}".format(len(findAttractors(W))))

def assignment3_2():
    X = loadData('pict.dat') #loads 32*32x11
    W = compW(X[:3,:],iter=10)

    exciteIm = X[10]

    recalledIms = recallAsync(exciteIm, W)

    plotIms(X, exciteIm, recalledIms, save=1, append="", path="")

    #recalledIm = np.array(recall(exciteIm, W))


if __name__ == '__main__':
    #assignment3_0()
    #assignment3_1()
    assignment3_2()
