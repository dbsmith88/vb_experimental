import collections


class SortedHeap:
    def __init__(self, n, target=1.0):
        self.n = n
        self.current_n = 0
        self.target = target
        self.max = None
        self.min = None
        v = float("inf")
        self.s_heap = [[v, None]] * self.n

    def purge(self):
        v = float("inf")
        del self.s_heap
        self.s_heap = [[v, None]] * self.n

    def add(self, data, metric):
        # Check if data already exists in s_heap
        if metric == float("inf"):
            return
        for i in self.s_heap:
            if i[1] is not None:
                if collections.Counter(i[1].attributes) == collections.Counter(data.attributes):
                    return
        metric = abs(self.target - abs(metric))
        if self.current_n == 0:
            self.s_heap[0] = [metric, data]
            self.max = metric
            self.min = metric
        elif self.current_n < self.n or metric < self.max:
            self.s_heap[self.n - 1] = [metric, data]
            self.s_heap = sorted(self.s_heap, key=lambda x: x[0])
        self.current_n = sum(i[1] is not None for i in self.s_heap)
        self.max = max(self.s_heap, key=lambda x: x[0])[0]
        self.min = min(self.s_heap, key=lambda x: x[0])[0]

    def to_list(self):
        s_heap_list = []
        for i in self.s_heap:
            s_heap_list.append(i[1])
        return s_heap_list

    def get_top(self, n=1):
        s_heap_list = []
        n = n if n <= self.n else self.n
        for i in range(0, n):
            s_heap_list.append(list(self.s_heap[i][1].attributes))
        return s_heap_list

    def get_bottom(self, n=1):
        s_heap_list = []
        n = n if n <= self.n else self.n
        low_n = self.current_n - n - 1 if self.current_n - n - 1 > 0 else 0
        for i in range(self.current_n - 1, low_n, -1):
            s_heap_list.append(list(self.s_heap[i][1].attributes))
        return s_heap_list

    def print(self, metric="metric"):
        print("\nRank, {} and model attribute list".format(metric.upper()))
        for i, v in enumerate(self.s_heap):
            if v[1] is not None:
                print("{}: {}\t-\t{}".format(i+1, v[0], ", ".join(list(v[1].attributes))))
