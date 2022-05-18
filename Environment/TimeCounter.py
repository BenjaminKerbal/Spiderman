import time

''' Used for speed optimization '''

class TimeInstance:

    def __init__(self):
        self.latest_time = time.time()
        self.sum_time = 0
        self.counter = 0

    def new_time_count(self):
        self.counter += 1
        new_time = time.time()
        self.sum_time += new_time - self.latest_time
        self.latest_time = new_time

    def start_timer(self):
        self.latest_time = time.time()

    def end_timer(self):
        self.counter += 1
        self.sum_time += time.time() - self.latest_time

    def get_total_time(self):
        return self.sum_time

    def get_mean_time(self):
        return self.sum_time / self.counter


class TimeCounter:

    __Instance = None

    @classmethod
    def getInstance(cls):
        if cls.__Instance is None:
            cls.__Instance = cls()
        return cls.__Instance

    def __init__(self):
        self.times = {}

    def time_between_calls(self, text):
        if not text in self.times:
            self.times[text] = TimeInstance()
        else:
            self.times[text].new_time_count()
    
    def time_count_selection(self, text, start_new_time : bool):
        if not text in self.times:
            self.times[text] = TimeInstance()

        if start_new_time:
            self.times[text].start_timer()
        else:
            self.times[text].end_timer()

    def print_stats(self):
        print(f'{"Time name":9} : {"mean":5} : {"total":5}')
        for key, value in self.times.items():
            print(f'{key:9} : {round(value.get_mean_time()*1000, 5):5} ms : {round(value.get_total_time(), 5):5} s')