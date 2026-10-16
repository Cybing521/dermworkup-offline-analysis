from .fields import present


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        if not present(a) or not present(b):
            return
        ra, rb = self.find(str(a)), self.find(str(b))
        if ra != rb:
            self.parent[rb] = ra
