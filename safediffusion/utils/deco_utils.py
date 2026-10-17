import time

def time_profile(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        print(f"[Time Start]        Function '{func.__name__}'")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[Time   End]        Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    
    return wrapper