class DisjointSet:
    def __init__(self,size):
        self.parent = range(size)
        self.rank = [0]*size

    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        px,py = self.find(x),self.find(y)
        if px==py:
            return False
        elif self.rank[px] > self.rank[py]:
            self.parent[py]=px
        elif self.rank[px] < self.rank[py]:
            self.parent[px]=py
        else:
            self.parent[py] = px
            self.rank[px]+=1
        return True

class DisjointSetLight:
    def __init__(self, size):
        self.parent = range(size)

    def find(self, x):
        if self.parent[x]!= x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x),self.find(y)
        if px != py:
            self.parent[py] = px
