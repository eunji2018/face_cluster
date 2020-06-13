import numpy
from time import time
import scipy.io as scio
from queue import Queue
from evaluate import evaluate\

file_path = 'D:\workspace\\face_verification_experiment\\results\LightenedCNN_C_lfw.mat'
data = scio.loadmat(file_path)
sample = data['features']
labels = data['labels'][0]

n_sample = sample.shape[0]
n_feature = sample.shape[1]

# 进行聚类
def cluster(threshold, k):
    # distance = numpy.zeros((n_sample, n_sample))
    # for i in range(n_sample):
    #     distance[:, i] = numpy.linalg.norm(sample - sample[i], axis=1)
    #
    # nn_list = build_list(distance)
    # rank_order_dist = rank_order_distance(nn_list, k)
    #
    # numpy.savetxt('face_rank_order\\face_rank_order_dist_1_210.txt', rank_order_dist)

    rank_order_dist = numpy.loadtxt('face_rank_order\\face_rank_order_dist_1_200.txt')

    cluster_result = numpy.zeros(n_sample)
    flag = numpy.zeros(n_sample)
    count = 0
    cluster = 0
    queue = Queue()
    while count < n_sample:
        for i in range(n_sample):
            if flag[i] == 0:
                break
        queue.put(i)
        cluster_result[i] = cluster
        flag[i] = 1
        count += 1
        while not queue.empty():
            point = queue.get()
            for j in range(n_sample):
                if rank_order_dist[point, j] <= threshold and flag[j] == 0:
                    queue.put(j)
                    cluster_result[j] = cluster
                    flag[j] = 1
                    count += 1

        cluster += 1

    print('clusters:{}'.format(cluster))
    return cluster_result


# 构造近邻列表
def build_list(distance):
    nn_list = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        nn_list[i] = numpy.argsort(distance[i])

    return nn_list

# 计算rank-order距离
def rank_order_distance(nn_list, k):
    rank_order_dist = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            rank_order_dist[i, j] = float('inf')

    for i in range(n_sample):
        print(i)
        for j in range(i+1, n_sample):
            index_i = numpy.argwhere(nn_list[i] == j)[0][0]
            index_j = numpy.argwhere(nn_list[j] == i)[0][0]
            if index_i <= k or index_j <= k:
                distance_i = 0
                distance_j = 0
                for r in range(min(index_i, k) + 1):
                    distance_i += numpy.argwhere(nn_list[j] == nn_list[i, r])[0][0]
                for r in range(min(index_j, k) + 1):
                    distance_j += numpy.argwhere(nn_list[i] == nn_list[j, r])[0][0]
                # 对称化距离
                rank_order_dist[i ,j] = rank_order_dist[j, i] = (distance_i + distance_j) / min(index_i, index_j)

    return rank_order_dist

# 最佳结果：Rank-Order距离阈值95，时间x，F值x，簇的个数x
if __name__ == '__main__':
    start_time = time()
    result = cluster(95, 200)
    time_diff = time() - start_time
    print('time:{}'.format(time_diff))
    evaluate(labels, result)

