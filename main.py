# import numpy as np
# import time
# from multiprocessing import Pool
# import matplotlib.pyplot as plt

# def multiply_matrices(seed):
#     np.random.seed(seed)
#     A = np.random.rand(100, 100)
#     B = np.random.rand(100, 100)
#     return np.dot(A, B)

# if __name__ == '__main__':
#     num_matrices = 100
#     num_threads_range = range(1, 9)  # Range of number of threads
#     time_taken = []  # To store time taken for each number of threads

#     for num_threads in num_threads_range:
#         start_time = time.time()
#         with Pool(num_threads) as p:
#             results = p.map(multiply_matrices, range(num_matrices))
#         end_time = time.time()
#         total_time = end_time - start_time
#         time_taken.append(total_time)

#         print(f"T = {num_threads}\t\t{total_time:.2f}")

#     # Generate the table
#     print("\nThreads\t\tTime Taken(sec)")
#     for i in range(len(num_threads_range)):
#         print(f"T = {num_threads_range[i]}\t\t{time_taken[i]:.2f}")

#     # Generate the graph
#     plt.plot(num_threads_range, time_taken, marker='o')
#     plt.xlabel('Threads')
#     plt.ylabel('Time Taken (sec)')
#     plt.title('Number of Threads vs Time Taken')
#     plt.grid(True)
#     plt.show()



import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

def multiply_matrices(seed):
    np.random.seed(seed)
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    return np.dot(A, B)

def task(num_matrices, num_threads):
    start_time = time.time()
    with Pool(num_threads) as p:
        results = p.map(multiply_matrices, range(num_matrices))
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

if __name__ == '__main__':
    startTime = time.time()
    num_matrices = 100
    num_threads_range = range(1, 9)  # Range of number of threads
    time_taken = []  # To store time taken for each number of threads

    print("Program Started....")

    # Main loop to run the task function with different numbers of threads
    for num_threads in num_threads_range:
        total_time = task(num_matrices, num_threads)
        time_taken.append(total_time)

        print(f"T = {num_threads}\t\t{total_time:.2f}")

    # Generate the table
    print("\nThreads\t\tTime Taken(sec)")
    for i in range(len(num_threads_range)):
        print(f"T = {num_threads_range[i]}\t\t{time_taken[i]:.2f}")

    # Generate the graph
    plt.plot(num_threads_range, time_taken, marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time Taken (sec)')
    plt.title('Number of Threads vs Time Taken')
    plt.grid(True)
    plt.show()

    print("Total Time %f sec" % (round(time.time() - startTime, 4)))
