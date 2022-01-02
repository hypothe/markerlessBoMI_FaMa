import time


class StopWatch:
    def __init__(self):
        self._start_time = 0
        self._elapsed_time = 0
        self._pause_time = 0
        self._elapsed_pause = 0

    def start(self):
        self._start_time = time.perf_counter_ns() * 10**-6

    @property
    def start_time(self):
        return self._start_time

    # @property
    # def elapsed_time(self):
    #     return self._elapsed_time

    @property
    def elapsed_time(self):
        self._elapsed_time = time.perf_counter_ns() * 10 ** -6 - self._start_time - self._elapsed_pause
        return self._elapsed_time

    @property
    def elapsed_pause(self):
        return self._elapsed_pause

    def pause(self):
        self._pause_time = time.perf_counter_ns() * 10 ** -6

    def restart(self):
        self._elapsed_pause += (time.perf_counter_ns() * 10 ** -6 - self._pause_time)
