class UnionFind:
    """
    Union-Find data structure implementation for tracking equivalent hint sets
    """
    def __init__(self, n):
        """Initialize with n elements"""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find the root/representative of element x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union the sets containing x and y using rank heuristic"""
        xroot = self.find(x)
        yroot = self.find(y)

        if xroot == yroot:
            return
        
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
    
    def get_elements_in_set(self, x):
        """Get all elements in the same set as x"""
        xroot = self.find(x)
        return [i for i in range(len(self.parent)) if self.find(i) == xroot] 