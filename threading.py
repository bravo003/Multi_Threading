import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import pandas as pd

# Function to perform matrix multiplication and measure time
def matrix_multiplication_with_threads(constant_matrix, num_threads):
    time_results = []
    for _ in range(10):
        start_time = time.time()
        threads = []
        batch_size = 100 // num_threads
        for i in range(0, 100, batch_size):
            thread = threading.Thread(target=multiply_batch, args=(constant_matrix, time_results, i, min(i+batch_size, 100)))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        end_time = time.time()
        time_results.append(end_time - start_time)
    return np.mean(time_results)

# Function to perform matrix multiplication for a batch of matrices
def multiply_batch(constant_matrix, time_results, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        random_matrix = np.random.rand(1000, 1000)
        result = np.matmul(random_matrix, constant_matrix)

# Create a constant matrix
constant_matrix = np.random.rand(1000, 1000)

# Generate a table for time taken with different numbers of threads
thread_counts = list(range(1, 11))
time_taken = []
for num_threads in thread_counts:
    avg_time = matrix_multiplication_with_threads(constant_matrix, num_threads)
    time_taken.append(avg_time)
    print(f"Time taken with {num_threads} threads: {avg_time} seconds")

# Plot the results
plt.plot(thread_counts, time_taken)
plt.xlabel('Number of Threads')
plt.ylabel('Average Time (seconds)')
plt.title('Time Taken vs Number of Threads')
plt.show()

# Save the results to a CSV file
data = {'Number of Threads': thread_counts, 'Average Time (seconds)': time_taken}
df = pd.DataFrame(data)
df.to_csv('matrix_multiplication_results.csv', index=False)
