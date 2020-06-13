import numpy
from rank_order_1 import cluster1
from rank_order_2 import cluster2
from rank_order_3 import cluster3
from evaluate import evaluate

array = numpy.zeros(3)

def compare():
    sample_1 = numpy.random.randn(100, 2) + [5, 5]
    sample_2 = numpy.random.randn(100, 2) + [11, 5]
    sample_3 = numpy.random.randn(100, 2) + [8, 10]
    sample = numpy.concatenate((sample_1, sample_2, sample_3), axis=0)
    label = []
    for i in range(3):
        for j in range(100):
            label.append(i)

    result = cluster1(sample, 29, 10)
    score_1 = evaluate(label, result)
    print('---')
    result = cluster2(sample, 1, 10)
    score_2 = evaluate(label, result)
    print('---')
    result = cluster3(sample, 4, 10)
    score_3 = evaluate(label, result)
    max_score = max(score_1, score_2, score_3)

    if score_1 == max_score:
        array[0] += 1
    if score_2 == max_score:
        array[1] += 1
    if score_3 == max_score:
        array[2] += 1

if __name__ == '__main__':
   for i in range(10000):
       compare()
       print(i)
   print(array)

