import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        data = [line.split(',') for line in lines]
        data = np.asarray([int(i) for i in data[0]])
    return data

def sign(x):
    if x<0:
        return -1
    else:
        return 1

def sign_binary(x):
    if x<=0:
        return 0
    else:
        return 1

def plot_recall(no_noise, noisy, recall):
        plt.figure()
        plt.subplot(131)
        plt.imshow(no_noise.reshape(32,32))
        plt.title("No noise")
        plt.subplot(132)
        plt.imshow(noisy.reshape(32,32))
        plt.title("Noisy")
        plt.subplot(133)
        plt.imshow(recall.reshape(32,32))
        plt.title("Recalled")

def generate_random_patterns(size, n_patterns, pattern_type="bipolar", bias=0):
    rand_patterns = []
    if pattern_type=="bipolar":
        for i in range(n_patterns):
            rand_patterns.append(np.squeeze(np.random.choice([-1, 1], size))+bias)
    else:
        for i in range(n_patterns):
            rand_patterns.append(np.squeeze(np.random.choice([0, 1], size))+bias)
    rand_patterns = np.asarray(rand_patterns)
    return rand_patterns

def learn_patterns(patterns):
    N = len(patterns[0])
    W = np.zeros((N,N))
    for pattern in patterns:
        W += np.outer(pattern, pattern)
    return W

def learn_pattern_sparse(pattern, activity, scale):
    # Learns only one pattern at a time
    N = len(pattern)
    di = np.diag_indices(N)
    W = np.outer(pattern-activity, pattern-activity)
    # W[di] = 0
    return W*scale

def network_recall_sync(pattern, W, attractor):
    local_pattern = np.copy(pattern)
    count = 0
    old_energy = energy_func(W, local_pattern)
    im_size = int(np.sqrt(len(pattern)))
    vect_sign = np.vectorize(sign)
    # plt.figure()
    # plt.imshow(local_pattern.reshape(im_size, im_size))
    # plt.show()
    while True:
        local_pattern = vect_sign((W @ local_pattern))
        energy = energy_func(W, local_pattern)
        # print(energy)
        # plt.figure()
        # plt.imshow(local_pattern.reshape(im_size, im_size))
        # plt.show()
        if energy == old_energy:
            # print("Number of iterations for stability: {}".format(count))
            # print("Energy at stability: {}".format(energy))
            return local_pattern, count
        old_energy = energy
        count += 1

def network_recall_async(pattern, W, attractor):
    local_pattern = np.copy(pattern)
    count = 1
    energy_att = energy_func(W, attractor)
    while True:
        energy = energy_func(W, local_pattern)
        print(energy)
        if count%100==0:
            print(count)
            plt.figure()
            plt.imshow(local_pattern.reshape(32,32))
            plt.show()

        idx = np.random.randint(len(local_pattern))
        local_pattern[idx] = np.sign(np.dot(W[idx,:], local_pattern))

        if energy == energy_att:
            print("Number of iterations for stability: {}".format(count))
            return local_pattern
        count += 1

def sparse_recall_sync(pattern, attractor, W, theta):
    local_pattern = np.copy(pattern)
    count = 0
    att_energy = energy_func(W, attractor)
    im_size = int(np.sqrt(len(pattern)))
    vect_sign = np.vectorize(sign)
    while True:
        local_pattern = 0.5 + 0.5*vect_sign((W @ local_pattern) - theta)
        energy = energy_func(W, local_pattern)
        if energy == att_energy:
            # print("Number of iterations for stability: {}".format(count))
            # print("Energy at stability: {}".format(energy))
            return local_pattern, count
        elif count == 10:
            # print("Not converged")
            return None, count
        count += 1

def energy_func(W, pattern):
    E = -np.dot(np.dot(pattern, W), pattern)
    return E

def distort_pattern(pattern, change, n_noise):
    local_pattern = np.copy(pattern)
    local_pattern[np.random.permutation(range(len(local_pattern)))[:n_noise]]*=change
    return local_pattern

if __name__ == '__main__':

    """Convergence and attractors"""
    # patterns = [[-1, -1, 1, -1, 1, -1, -1, 1],
    #             [-1, -1, -1, -1, -1, 1, -1, -1],
    #             [-1, 1, 1, -1, -1, 1, -1, 1]]
    # W = learn_patterns(patterns)
    #
    #
    # x1d=[1, -1, 1, -1, 1, -1, -1, 1]
    # x2d=[1, 1, -1, -1, -1, 1, -1, -1]
    # x3d=[1, 1, 1, -1, 1, 1, -1, 1]
    #
    # recall_x1d = network_recall(x1d, W)
    # recall_x2d = network_recall(x2d, W)
    # recall_x3d = network_recall(x3d, W)
    #
    # print("Recalled: {}, Stored: {}, x1d==patterns[0]=={}".format(recall_x1d, patterns[0], np.all(recall_x1d==patterns[0])))
    # print("Recalled: {}, Stored: {}, x2d==patterns[1]=={}".format(recall_x2d, patterns[1], np.all(recall_x2d==patterns[1])))
    # print("Recalled: {}, Stored: {}, x3d==patterns[2]=={}".format(recall_x3d, patterns[2], np.all(recall_x3d==patterns[2])))

    """Sequential update & Energy"""
    # data = read_data("pict.dat")
    # p1 = data[0:1024]
    # p2 = data[1024:2048]
    # p3 = data[2048:3072]
    # p10 = data[9216:10240]
    # p11 = data[10240:11264]
    # patterns = [p1, p2, p3]
    # W_seq = learn_patterns(patterns)

    # W_rand = np.random.normal(0, 0.1, size=W_seq.shape)
    # W_rand = 0.5*(W_rand+W_rand.T)
    # p1_recall = network_recall_sync(p1, W_seq, p1)
    # p2_recall = network_recall_sync(p2, W_seq)
    # p3_recall = network_recall_sync(p3, W_seq)
    # p10_recall = network_recall_sync(p2, W_seq, p1)
    # p11_recall_sync = network_recall_sync(p11, W_seq)
    # p11_recall_async = network_recall_async(p11, W_seq)
    # p10_recall_async = network_recall_async(p10, W_rand, p1)
    # p11_recall_async = network_recall_async(p11, W_seq, p3)

    # energy_p1 = energy_func(W_seq, p11)
    # energy_p2 = energy_func(W_seq, p2)
    # energy_p3 = energy_func(W_seq, p3)
    # print(energy_p1)
    # print(energy_p2)
    # print(energy_p3)

    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(p10.reshape(32,32))
    # plt.title("Input")
    # plt.subplot(132)
    # plt.imshow(p10_recall.reshape(32,32))
    # plt.title("Recalled image")
    # plt.subplot(133)
    # plt.imshow(p1.reshape(32,32))
    # plt.title("Stored image")
    # plt.show()

    """Distortion resistance"""
    # data = read_data("pict.dat")
    # p1 = data[0:1024]
    # p2 = data[1024:2048]
    # p3 = data[2048:3072]
    # p4 = data[3072:4096]
    # p5 = data[4096:5120]
    # p6 = data[5120:6144]
    # patterns = [p1, p2, p3, p6]
    # W = learn_patterns(patterns)
    # n_noise=1024*0
    #
    # p1_distort = distort_pattern(p1, int(n_noise))
    # # p2_distort = distort_pattern(p2, int(n_noise))
    # # p3_distort = distort_pattern(p3, int(n_noise))
    # p1_recall = network_recall_sync(p1_distort, W, p1)
    # # p2_recall = network_recall_sync(p2_distort, W, p2)
    # # p3_recall = network_recall_sync(p3_distort, W, p3)
    # plot_recall(p1, p1_distort, p1_recall)
    # # plot_recall(p3, p3_distort, p3_recall)
    # # plot_recall(p2, p2_distort, p2_recall)
    # plt.show()

    """Random patterns"""
    # patterns = generate_random_patterns(100, 300)
    # sign = np.vectorize(sign)
    # N = len(patterns[0])
    # W_rand = np.zeros((N,N))
    # di = np.diag_indices(N)
    # n_noise = 5
    # stable_patterns_percent = []
    # stable_patterns_percent_noise = []
    # for i, pattern in enumerate(patterns):
    #     if i%10 == 0:
    #         print("Learning pattern: {}".format(i))
    #     W_rand += np.outer(pattern, pattern)
    #     W_rand[di] = 0
    #     W_rand *= 1/N
    #     stable = 0
    #     stable_noise = 0
    #     for j in range(i+1):
    #         # print("Recall pattern no: {}".format(j))
    #         noisy_pat = distort_pattern(patterns[j], n_noise)
    #         recall_noise, count_noise = network_recall_sync(noisy_pat, W_rand, patterns[j])
    #         recall, count= network_recall_sync(patterns[j], W_rand, patterns[j])
    #         if count == 0:
    #             stable+=1
    #         if count_noise == 0:
    #             stable_noise +=1
    #     stable_patterns_percent.append(stable/(i+1))
    #     stable_patterns_percent_noise.append(stable_noise/(i+1))
    #
    # plt.figure()
    # plt.plot(stable_patterns_percent)
    # plt.title("Clean patterns")
    # plt.xlabel("Number of stored patterns")
    # plt.ylabel("Percentage of stable patterns")
    # plt.savefig("clean_patterns_0_diag.png", dpi="figure", format="png")
    # plt.figure()
    # plt.plot(stable_patterns_percent_noise)
    # plt.title("Noisy patterns, {} flipped units".format(n_noise))
    # plt.xlabel("Number of stored patterns")
    # plt.ylabel("Percentage of stable patterns")
    # plt.savefig("noisy_patterns_{}_flipped_units_0_diag.png".format(n_noise), dpi="figure", format="png")
    # plt.show()

    """Sparse patterns"""
    # patterns = np.ones((300, 100))
    # patterns_sparse = []
    # N = len(patterns[0])
    # thetas = [0.001, 0.01, 0.1, 1, 2, 4, 8, 10, 15, 20]
    # activity = 0.01
    # n_stored = []
    # for pattern in patterns:
    #     patterns_sparse.append(distort_pattern(pattern, 0, int(N*(1-activity))))
    # # print(patterns_sparse[0])
    # # W = learn_pattern_sparse(patterns_sparse[0], activity, 1)
    # # recall, count = sparse_recall_sync(patterns_sparse[0], patterns_sparse[0], W, 1)
    # # print(recall)
    # W = np.zeros((N,N))
    # for pattern in patterns_sparse:
    #     W += learn_pattern_sparse(pattern, activity, 1)
    # for theta in thetas:
    #     stored_OK = 0
    #     for i, pattern in enumerate(patterns_sparse):
    #         recall, count = sparse_recall_sync(pattern, pattern, W, theta)
    #         if recall is not None:
    #             stored_OK += 1
    #     n_stored.append(stored_OK)
    # print(n_stored)
    # x = np.arange(len(thetas))
    # plt.figure()
    # plt.bar(x, n_stored, align="edge", width=0.3)
    # plt.xticks(x, thetas)
    # plt.ylim((0, 300))
    # plt.ylabel("Number of patterns stored")
    # plt.xlabel("Theta")
    # plt.title("Number of patterns stored, activity: {}".format(activity))
    # plt.savefig("sparse_p_{}_theta_0.001-20.png".format(activity), dpi="figure", format="png")
    # plt.show()
