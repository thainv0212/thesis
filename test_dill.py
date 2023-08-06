import dill
import json
import time
import pickle
# path = 'ice/logdata_20220706094108.txt'
# data = json.load(open(path))
# dill.dump(data, open('dump_dill.pkl', 'wb'))
# pickle.dump(data, open('dump_pickle.pkl', 'wb'))
# start = time.time()
# for _ in range(100):
#     data = json.load(open(path))
# end = time.time()
# print('time for json', (end - start) * 1000/100)
#
# start = time.time()
# for _ in range(100):
#     data = dill.load(open('dump_dill.pkl', 'rb'))
# end = time.time()
# print('time for dill', (end - start) * 1000/100)

start = time.time()
for _ in range(100):
    data = pickle.load(open('0.pkl', 'rb'))
end = time.time()
print('time for dill', (end - start) * 1000/100)

