import numpy
import matplotlib.pyplot as pyplot
import scipy.io as scio
from time import time
from queue import Queue
from evaluate import evaluate
from evaluate import build_list

file_path = 'D:\workspace\\face_verification_experiment\\results\LightenedCNN_C_lfw.mat'
data = scio.loadmat(file_path)
sample = data['features']
labels = data['labels'][0]

n_sample = sample.shape[0]
n_feature = sample.shape[1]

# 进行聚类：threshold相似度阈值
def cluster(threshold, k):
    # distance = numpy.zeros((n_sample, n_sample))
    # for i in range(n_sample):
    #     distance[:, i] = numpy.linalg.norm(sample - sample[i], axis=1)
    # nn_list = build_list(distance, n_sample)
    # rank_order_simi = rank_order_similarity(nn_list, k)
    #
    # numpy.savetxt('face_rank_order\\face_rank_order_simi_3_200.txt', rank_order_simi)

    rank_order_simi = numpy.loadtxt('face_rank_order\\face_rank_order_simi_3_200.txt')

    print('---')
    print(numpy.max(rank_order_simi))

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
                if rank_order_simi[point, j] >= threshold and flag[j] == 0:
                    queue.put(j)
                    cluster_result[j] = cluster
                    flag[j] = 1
                    count += 1

        cluster += 1

    print('clusters:{}'.format(cluster))
    return cluster_result


# 计算rank-order相似度
def rank_order_similarity(nn_list, k):
    rank_order_simi = numpy.zeros((n_sample, n_sample))

    for i in range(n_sample):
        print(i)
        for j in range(i+1, n_sample):
            index_i = numpy.argwhere(nn_list[i] == j)[0][0]
            index_j = numpy.argwhere(nn_list[j] == i)[0][0]
            if index_i <= k or index_j <= k:
                similarity_i = 0
                similarity_j = 0
                for r in range(min(index_i, k) + 1):
                    temp = numpy.argwhere(nn_list[j] == nn_list[i, r])[0][0]
                    if temp <= k:
                        for t in range(r+1):
                            if numpy.argwhere(nn_list[j] == nn_list[i, t])[0][0] <= temp:
                                similarity_i += 1
                for r in range(min(index_j, k) + 1):
                    temp = numpy.argwhere(nn_list[i] == nn_list[j, r])[0][0]
                    if temp <= k:
                        for t in range(r+1):
                            if numpy.argwhere(nn_list[i] == nn_list[j, t])[0][0] <= temp:
                                similarity_j += 1
                # 对称化距离
                rank_order_simi[i ,j] = rank_order_simi[j, i] = (similarity_i + similarity_j) / min(index_i, index_j)

    return rank_order_simi

if __name__ == '__main__':
    start_time = time()
    result = cluster(600, 200)
    time_diff = time() - start_time
    print('time:{}'.format(time_diff))
    evaluate(labels, result)

