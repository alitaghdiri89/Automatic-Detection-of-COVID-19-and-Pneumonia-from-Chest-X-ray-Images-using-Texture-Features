from db_creator import create_db
from classifier import classify
import winsound
import matplotlib.pyplot as plt


def print_results(best, mean):
    print('\nBEST:\n')
    print(best)
    print('\nMEAN:\n')
    print(mean)


MAKE_DATABASE = False
USE_SINGLE_K_VALUE = False
class_count = 3
single_k_value = 3
# clf type can be 'tree', 'random_forest', 'svm', 'knn' or 'gaussian_naive_bayes'
clf_type = 'gaussian_naive_bayes'
results = None

try:
    if MAKE_DATABASE:
        create_db(class_count)
    else:
        results = classify(clf_type, class_count)
        total_best = None
        for params, best_result, mean_result, mean_time in results:
            print('For parameters: ', params)
            print_results(best_result, mean_result)
            print('mean_time :', mean_time, 'seconds')
            print('*' * 60, '\n')
            if total_best is None or total_best[2] < mean_result:
                total_best = (params, best_result, mean_result, mean_time)
        print('*' * 50, '\n', '*' * 40, "\nBest Results: params =", total_best[0])
        print_results(total_best[1], total_best[2])
        print('mean_time :', total_best[3], 'seconds')
except Exception:
    raise
finally:
    winsound.Beep(frequency=440, duration=1000)
# plots, first accuracy vs K for KNN
if not MAKE_DATABASE and not USE_SINGLE_K_VALUE and clf_type == 'knn':
    k_values = list()
    accuracies = list()
    for params, best_result, mean_result, mean_time in results:
        k_values.append(params['n_neighbors'])
        accuracies.append(mean_result.accuracy)
    plt.plot(k_values, accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    max_acc_str = '%.2f' % (max(accuracies) * 100) + '%'
    plt_title = f"Maximum accuracy: {max_acc_str} at K = {k_values[accuracies.index(max(accuracies))]}"
    plt.title(plt_title)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
