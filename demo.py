import numpy

# sample = numpy.random.randint(10, size=(3, 3))
# print(sample)
# print(numpy.max(sample))
# print(numpy.min(sample))

# distance = numpy.zeros((3, 3))
# for i in range(3):
#     distance[:, i] = numpy.linalg.norm(sample - sample[i], axis=1)
#
# print(distance)

# print(numpy.argsort(list))
# temp = numpy.argsort(list)[:4]
# print(temp)
# array = numpy.zeros((2,3))
# numpy.savetxt('demo.txt', array)
# print(array)
# temp = numpy.loadtxt('demo.txt')
# print(temp)

# temp = numpy.zeros((5, 5))
# print(temp)
# temp[1, 2] = temp[2, 1] = 1
# print(temp)

# for i in range(5):
#     print('---')
# print(i)

rank_order_simi = numpy.loadtxt('face_rank_order\\face_rank_order_simi_3_200.txt')
print('---')
for i in range(len(rank_order_simi)):
    for j in range(len(rank_order_simi)):
        if rank_order_simi[i, j] >= 900:
            print(i, j)

# array = numpy.zeros(3)
# print(array)