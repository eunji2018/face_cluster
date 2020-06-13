import numpy
import matplotlib.pyplot as pyplot
from queue import Queue
from evaluate import evaluate
from evaluate import build_list

# 进行聚类
def cluster2(sample, threshold, k):
    n_sample = sample.shape[0]
    n_feature = sample.shape[1]

    distance = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        distance[:, i] = numpy.linalg.norm(sample - sample[i], axis=1)
    nn_list = build_list(distance, n_sample)
    rank_order_dist = rank_order_distance(nn_list, k, n_sample)

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

    # print('clusters:{}'.format(cluster))
    # centers = numpy.zeros((cluster, n_feature))
    # for i in range(cluster):
    #     centers[i] = numpy.mean(sample[cluster_result == i], axis=0)
    #
    # pyplot.clf()
    # pyplot.scatter(sample[:, 0], sample[:, 1], alpha=0.5, c=cluster_result)
    # pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', c='k')
    # pyplot.show()

    return cluster_result

# 计算rank-order距离
def rank_order_distance(nn_list, k, n_sample):
    rank_order_dist = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            rank_order_dist[i, j] = float('inf')

    for i in range(n_sample):
        for j in range(i+1, n_sample):
            index_i = numpy.argwhere(nn_list[i] == j)[0][0]
            index_j = numpy.argwhere(nn_list[j] == i)[0][0]
            if index_i <= k or index_j <= k:
                distance_i = 0
                distance_j = 0
                for r in range(min(index_i, k) + 1):
                    if numpy.argwhere(nn_list[j] == nn_list[i, r])[0][0] > k:
                        distance_i += 1
                for r in range(min(index_j, k) + 1):
                    if numpy.argwhere(nn_list[i] == nn_list[j, r])[0][0] > k:
                        distance_j += 1
                # 对称化距离
                rank_order_dist[i ,j] = rank_order_dist[j, i] = (distance_i + distance_j) / min(index_i, index_j)

    return rank_order_dist

