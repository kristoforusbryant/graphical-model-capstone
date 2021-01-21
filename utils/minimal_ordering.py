class Vertex: 
    def __init__(self, label,  i = 0):
        self.i = i
        self.label = label
        self.label.sort()
        self.label.reverse()
    def __eq__(self, other): 
        return self.label == other.label
    def __lt__(self, other): 
        for i in range(min(len(self.label), len(other.label))): 
            if self.label[i] < other.label[i]: return True
            if self.label[i] > other.label[i]: return False
        if len(self.label) < len(other.label): return True
        return False 
    def __gt__(self, other): 
        if self.label == other.label: return False
        return not self.__lt__(other)
    def __repr__(self): 
        return("(" + self.label.__str__() + ", " + self.i.__str__() + ")")

def DFS(dol, v, key, visited):
    DFSUtil(dol, v, visited)
    if key in visited: 
        return True 
    else: return False
    
def DFSUtil(dol, v, visited):
    visited.add(v) # adding in place
    for nb in dol[v]:
        if nb not in visited:
            DFSUtil(dol, nb, visited)
            
import copy
# Algorithm Implemented from (Rose and Tarjan, 1974) 
def LEXM(dol):
    n = len(dol)
    ordering = [] # vertices ordered from largest to smallest
    vertices = [Vertex([], i) for i in range(len(dol))] # list of labels
    for i in range(len(dol)): 
        vertices.sort()
        v = vertices.pop().i
        ordering.append(v)
        # DFS from every unnumbered vertex  
        vertices_ = copy.deepcopy(vertices) 
        for w in vertices_: 
            if w.i in dol[v]: 
                w.label.append(n-i-1)
            else: 
                visited = set([v_.i for v_ in vertices if (not v_ < w)]) 
                # removing vertices which labels are not less than w and not direct neighbour of w.
                visited = visited.union(set(ordering) - set([v]))
                if DFS(dol, w.i, v, visited): 
                    w.label.append(n-i-1)
        vertices = vertices_
    ordering.reverse()
    return ordering