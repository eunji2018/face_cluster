import numpy

# label表示真实标签，result表示聚类结果
def evaluate(label, result):
    class_pair = 0 # 所有类的样本对数量
    cluster_pair = 0 # 所有簇的样本对数量
    cluster_same = 0 # 划分到同一个簇的样本对数量
    class_same = 0 # 属于同一个类的样本对数量
    class_count = len(set(label)) # 类的个数
    cluster_count = len(set(result)) # 簇的个数

    for i in range(class_count):
        array = []
        for index in range(len(label)):
            if (label[index] == i):
                array.append(index)
        if (len(array) == 1):
            continue
        class_pair += len(array) * (len(array) - 1) / 2
        for j in range(len(array)):
            for k in range(j + 1, len(array)):
                if (result[array[j]] == result[array[k]]):
                    cluster_same += 1

    for i in range(cluster_count):
        array = []
        for index in range(len(result)):
            if (result[index] == i):
                array.append(index)
        if (len(array) == 1):
            continue
        cluster_pair += len(array) * (len(array) - 1) / 2
        for j in range(len(array)):
            for k in range(j + 1, len(array)):
                if (label[array[j]] == label[array[k]]):
                    class_same += 1

    # 计算precision、recall、f_score
    precision = class_same / cluster_pair
    recall = cluster_same / class_pair
    f_score = 2 * precision * recall / (precision + recall)

    print('class_pair:{}, cluster_pair:{}, cluster_same:{}, class_same:{}'
          .format(class_pair, cluster_pair, cluster_same, class_same))
    print('precision:{}, recall:{}, f_score:{}'.format(precision, recall, f_score))
    return f_score

# 构造近邻列表
def build_list(distance, n_sample):
    nn_list = numpy.zeros((n_sample, n_sample))
    for i in range(n_sample):
        nn_list[i] = numpy.argsort(distance[i])

    return nn_list