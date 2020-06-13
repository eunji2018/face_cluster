import numpy
import matplotlib.pyplot as pyplot
from evaluate import evaluate
from copy import deepcopy

sample_1 = numpy.random.randn(100, 2) + [5, 5]
sample_2 = numpy.random.randn(100, 2) + [9, 5]
sample_3 = numpy.random.randn(100, 2) + [7, 8]
sample = numpy.concatenate((sample_1, sample_2, sample_3), axis=0)
label = []
for i in range(3):
    for j in range(100):
        label.append(i)

n_sample = sample.shape[0]
n_feature = sample.shape[1]

# 生成初始质心
def generate_center(sample, k):
    f_mean = numpy.mean(sample, axis=0).reshape(1, n_feature)
    f_std = numpy.std(sample, axis=0).reshape(1, n_feature)
    centers = numpy.random.randn(k, n_feature) * f_std + f_mean
    return centers # 初始质心

# 进行聚类
def cluster(k):
    cluster_result = numpy.zeros(n_sample)
    distance = numpy.zeros((n_sample, k))
    centers_cur = generate_center(sample, k)
    centers_pre = numpy.zeros(centers_cur.shape)
    centers_diff = numpy.linalg.norm(centers_cur - centers_pre)

    epsilon = 0.001
    while centers_diff > epsilon:
        # 计算样本到质心的距离
        for i in range(k):
            distance[:, i] = numpy.linalg.norm(sample - centers_cur[i], axis=1)
        # 指派样本到最近的质心
        cluster_result = numpy.argmin(distance, axis=1)
        centers_pre = deepcopy(centers_cur)
        # 更新质心
        for i in range(k):
            temp = sample[cluster_result == i]
            centers_cur[i] = numpy.mean(temp, axis=0)
        centers_diff = numpy.linalg.norm(centers_cur - centers_pre)

    pyplot.clf()
    pyplot.scatter(sample[:, 0], sample[:, 1], alpha=0.5, c=cluster_result)
    pyplot.scatter(centers_cur[:, 0], centers_cur[:, 1], marker='*', c='k')
    pyplot.show()
    return cluster_result


if __name__ == '__main__':
    result = cluster(3)
    evaluate(label, result)
