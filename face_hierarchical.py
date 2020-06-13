import numpy
import matplotlib.pyplot as pyplot
import scipy.io as scio
from evaluate import evaluate
from queue import Queue
from time import time

file_path = 'D:\workspace\\face_verification_experiment\\results\LightenedCNN_C_lfw.mat'
data = scio.loadmat(file_path)
sample = data['features']
labels = data['labels'][0]

n_sample = sample.shape[0]
n_feature = sample.shape[1]


# 进行聚类
def cluster(threshold):
    distance = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        distance[:, i] = numpy.linalg.norm(sample - sample[i], axis=1)

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
                if distance[point, j] <= threshold and flag[j] == 0:
                    queue.put(j)
                    cluster_result[j] = cluster
                    flag[j] = 1
                    count += 1

        cluster += 1

    # centers = numpy.zeros((cluster+1, n_feature))
    # for i in range(cluster+1):
    #     centers[i] = numpy.mean(sample[cluster_result == i], axis=0)
    # pyplot.clf()
    # pyplot.scatter(sample[:, 0], sample[:, 1], alpha=0.5, c=cluster_result)
    # pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', c='k')
    # pyplot.show()
    print('clusters:{}'.format(cluster))
    return cluster_result

# 最佳结果：距离阈值108，时间306s，F值0.818，簇的个数9893
if __name__ == '__main__':
    start_time = time()
    result = cluster(108)
    time_diff = time() - start_time
    print('time:{}'.format(time_diff))
    evaluate(labels, result)

