import time
import resource
import json

MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = 0.5
NUM_CPUS = 8
NUM_GPUS = 4
FRAMES_PER_VIDEO = 5
NUM_VIDEOS = 160 # 160
NUM_FRAMES_TOTAL = FRAMES_PER_VIDEO * NUM_VIDEOS
FRAME_SIZE_B = 100 * MB
EXECUTION_MODE = "process"


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
