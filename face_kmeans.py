import scipy.io as scio
import numpy
import matplotlib.pyplot as pyplot
from evaluate import evaluate
from copy import deepcopy
from time import time

file_path = 'D:\workspace\\face_verification_experiment\\results\LightenedCNN_C_lfw.mat'
data = scio.loadmat(file_path)
sample = data['features']
labels = data['labels'][0]

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

    # pyplot.clf()
    # pyplot.scatter(sample[:, 0], sample[:, 1], alpha=0.5, c=cluster_result)
    # pyplot.scatter(centers_cur[:, 0], centers_cur[:, 1], marker='*', c='k')
    # pyplot.show()
    return cluster_result

# 最佳结果：K值2000，时间110s，F值0.175，簇的个数2000
if __name__ == '__main__':
    sum_time = 0
    sum_score = 0
    for i in range(10):
        start_time = time()
        result = cluster(2000)
        sum_time += time() - start_time
        sum_score += evaluate(labels, result)

    print('time:{}'.format(sum_time / 10))
    print('f_score:{}'.format(sum_score / 10))

