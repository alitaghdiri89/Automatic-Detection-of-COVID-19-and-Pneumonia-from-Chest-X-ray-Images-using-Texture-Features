import numpy as np

list_list_array = list()
for i in range(3):
    list_array = list()
    for j in range(3):
        array = np.random.rand(3)
        list_array.append(array)
    list_list_array.append(list_array)

arr3 = np.array(list_list_array)
print(arr3)
print('#######################3')
arr_sum = np.sum(arr3, axis=0)
print(arr_sum)
print('#######################3')
mean_arr = arr_sum / 3
print(mean_arr)
print(mean_arr[0])
print(type(mean_arr[0]))
