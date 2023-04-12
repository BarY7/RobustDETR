from time import perf_counter


class catchtime:

    def __init__(self, name):
        self.name = name
        self.time = None
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'
        print(f"Time of {self.name} is {self.readout} secs")