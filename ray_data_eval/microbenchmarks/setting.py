import time
import resource
import json
import os
import psutil
import multiprocessing
import sys

MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = 0.5
NUM_CPUS = 8
NUM_GPUS = 4
FRAMES_PER_VIDEO = 5
NUM_VIDEOS = 20 # 160
NUM_FRAMES_TOTAL = FRAMES_PER_VIDEO * NUM_VIDEOS
FRAME_SIZE_B = 100 * MB
EXECUTION_MODE = "process"

def log_memory_usage(mem_limit):
    
    tracker = MemoryTracker()
    tracker.log_memory_usage(mem_limit)

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.max_memory = 0

    def log_memory_usage(self, mem_limit):
        # Process-specific memory usage
        process_mem = self.process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        self.max_memory = max(self.max_memory, process_mem)

        # System-wide memory usage
        virtual_mem = psutil.virtual_memory()
        total_mem = virtual_mem.total / (1024 * 1024)  # Total memory in MB
        used_mem = virtual_mem.used / (1024 * 1024)    # Used memory in MB
        available_mem = virtual_mem.available / (1024 * 1024)  # Available memory in MB
        memory_percent = virtual_mem.percent           # Memory usage percentage

        print(f"  Process Memory Usage: {process_mem:.2f} MB")
        print(f"  Total Memory: {total_mem:.2f} MB")
        print(f"  Used Memory: {used_mem:.2f} MB ({memory_percent:.2f}%)")
        print(f"  Available Memory: {available_mem:.2f} MB\n")
        
        if used_mem > mem_limit * 1024:
            print("Memory exceeded!")

    def get_max_memory_usage(self):
        return self.max_memory

# Function to run memory logging in a process
def log_memory_usage_process(interval=2, mem_limit=10):
    process = psutil.Process(os.getpid())  # Get the current process (Spark driver)
    
    try:
        while True:
            log_memory_usage(mem_limit)
            time.sleep(interval)  # Log memory every `interval` seconds
    except KeyboardInterrupt:
        print("Memory tracking interrupted, shutting down.")
        return

def busy_loop(time_in_s):
    end_time = time.time() + time_in_s
    while time.time() < end_time:
        pass

def limit_cpu_memory(mem_limit):
    # limit cpu memory with resources
    mem_limit_bytes = mem_limit * GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))

def append_dict_to_file(data: dict, file_path: str):
    """
    Append a dictionary to a file as a JSON object.
    
    Args:
        data (dict): The dictionary to append.
        file_path (str): The path to the file.
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")
    
    # Open the file in append mode, creating it if it doesn't exist
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')
