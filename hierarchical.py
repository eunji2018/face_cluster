import numpy
import matplotlib.pyplot as pyplot
from evaluate import evaluate
from queue import Queue

sample_1 = numpy.random.randn(100, 2) + [5, 5]
sample_2 = numpy.random.randn(100, 2) + [11, 5]
sample_3 = numpy.random.randn(100, 2) + [8, 10]
sample = numpy.concatenate((sample_1, sample_2, sample_3), axis=0)
label = []
for i in range(3):
    for j in range(100):
        label.append(i)

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

    print('clusters:{}'.format(cluster))
    centers = numpy.zeros((cluster, n_feature))
    for i in range(cluster):
        centers[i] = numpy.mean(sample[cluster_result == i], axis=0)

    pyplot.clf()
    pyplot.scatter(sample[:, 0], sample[:, 1], alpha=0.5, c=cluster_result)
    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', c='k')
    pyplot.show()

    return cluster_result


if __name__ == '__main__':
    result = cluster(1)
    evaluate(label, result)
