import time


def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def timer():
    return time.time()


def log(msg):
    print(f"[LOG] {msg}")
