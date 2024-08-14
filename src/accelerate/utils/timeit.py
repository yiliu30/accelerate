import time

def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f


import time

class Timer:
    def __init__(self, msg="", time_record=None):
        self.msg = msg
        if time_record is None:
            time_record = TimeRecorder(f"{msg} time recorder")
        self.time_record = time_record
    
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        execution_time = round(execution_time * 1000)
        print(self.msg + f" time: {execution_time} ms")
        self.time_record.add_time(execution_time)


class TimeRecorder:
    def __init__(self, name=""):
        self.time_lst = []
        self.name = name

    def add_time(self, timer):
        self.time_lst.append(timer)
        return self
    
    def get_avg_time(self):
        return sum(self.time_lst) / len(self.time_lst)
    
    def get_total_time(self):
        return sum(self.time_lst)
    
    def __repr__(self) -> str:
        return f"{self.name} : total: {self.get_total_time()} ms, avg: {self.get_avg_time()} ms"

