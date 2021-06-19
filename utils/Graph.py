from typing import Set
import networkx as nx
import warnings
import numpy as np
import copy
from utils.minimal_ordering import LEXM
from utils.prime_components import primecomps


class Graph: # undirected, unweigheted graph
    def  __init__(self, n, dol=None):
        if dol is not None:
            if len(dol) != n:
                warnings.warn("n != len(dol), n is ignored")
            self._dol = dol
        else:
            self._dol = {i: [] for i in range(n)}
        self._name = ''

    # Getting and Printing
    def GetSize(self):
        return len(self._dol)
    def GetDOL(self):
        return self._dol
    def GetAdjM(self):
        AdjM = np.zeros((len(self._dol), len(self._dol)), dtype=int)
        for k,l in self._dol.items():
            for i in l:
                AdjM[i,k] = 1
                AdjM[k,i] = 1
        return AdjM
    def GetName(self):
        return self._name

    def GetID(self):
        return ''.join(self.GetBinaryL().astype(str).tolist())

    def GetBinaryL(self):
        n = len(self)
        binl = np.zeros(n * (n - 1) // 2, dtype=int)
        for i, l in self._dol.items():
            for j in l:
                if j < i: continue
                linear_idx = (n*(n-1)//2) - (n-i)*((n-i)-1)//2 + j - i - 1
                binl[linear_idx] = 1
        return binl

    def GetEdgeL(self):
        EdgeL = []
        for i in range(len(self._dol)):
            for j in range(i+1, len(self._dol)):
                if i in self._dol[j]:
                    EdgeL.append((i,j))
        return EdgeL

    def GetNeighbours(self, v):
        return self._dol[v]

    def IsEdge(self, i, j):
        if i < len(self) and j < len(self):
            return (j in self._dol[i])
        else:
            return None
    def EdgeCount(self):
        count = 0
        for i in range(len(self._dol)):
            for j in range(i+1, len(self._dol)):
                if i in self._dol[j]:
                    count += 1
        return count

    def _sort_dol(self, dol):
        for k in dol.keys():
            dol[k] = sorted(dol[k])
        return dol

    def __repr__(self):
        return self._dol.__str__()
    def __str__(self):
        return self._dol.__str__()
    def __len__(self):
        return len(self._dol)
    def __eq__(self, other):
        # TODO: can we ensure that dol is always sorted?
        return self._sort_dol(self._dol) == self._sort_dol(other._dol)
    def copy(self):
        return copy.deepcopy(self)

    def Draw(self, how=None, **kwargs):
        if how == 'circle':
            pos = self.GetCirclePos()
            nx.draw(nx.from_dict_of_lists(self._dol), with_labels=True, pos=pos, **kwargs)
        else:
            nx.draw(nx.from_dict_of_lists(self._dol), with_labels=True, **kwargs)
    def GetCirclePos(self):
        theta = 2 * np.pi / len(self._dol)
        keys = list(self._dol.keys())
        return {keys[i]: [-np.cos(theta * i), np.sin(theta * i)] for i in range(len(self._dol))}

    # Manually Setting
    def SetFromG(self, other):
        self._dol = other._dol
    def SetName(self, name):
        self._name = name

    def SetFromDOL(self,dol):
        self._dol = dol

    def SetFromAdjM(self, AdjM):
        AdjM = np.array(AdjM)
        self._dol = {i:[] for i in range(AdjM.shape[0])}
        for i in range(AdjM.shape[0]):
            for j in  range(i+1, (AdjM.shape[0])):
                if AdjM[i,j]:
                    self._dol[i].append(j)
                    self._dol[j].append(i)

    def SetFromString(self, s):
        A = np.zeros((len(self), len(self)))
        triu = np.triu_indices(len(self), 1)
        A[triu] = np.array(list(s)).astype(int)
        self.SetFromAdjM(A)

    def SetFromEdgeL(self, EdgeL):
        self.SetEmpty()
        self.AddEdges(EdgeL)

    def SetEmpty(self):
        self._dol = {i: [] for i in range(len(self))}

    def SetComplete(self, n=None):
        if n is None:
            n = len(self._dol)
        self._dol = {i: list(set(range(n)) - set([i])) for i in range(n)}

    def SetRandom(self):
        n = self.GetSize()
        a = np.random.randint(0, 2, n * (n-1) // 2)
        AdjM = np.zeros((n,n))
        AdjM[np.triu_indices(n, 1)] = a
        AdjM += np.transpose(AdjM)
        self.SetFromAdjM(AdjM)

    # Adding and Removing Edges
    def AddEdge(self,i,j):
        if j not in self._dol[i]:
            self._dol[i].append(j)
            self._dol[j].append(i)
    def AddEdges(self, lot):
        for i,j in lot:
            self.AddEdge(i,j)
    def RemoveEdge(self,i,j):
        if (j in self._dol[i]) and (i in self._dol[j]):
            self._dol[i].remove(j)
            self._dol[j].remove(i)
        else:
            raise ValueError("Edge does not exist")
    def RemoveEdges(self, lot):
        for i,j in lot:
            self.RemoveEdge(i,j)
    def FlipEdge(self, i,j):
        if j in self._dol[i]:
            self.RemoveEdge(i,j)
        else:
            self.AddEdge(i,j)
    def FlipEdges(self, lot):
        for i,j in lot:
            self.FlipEdge(i,j)

    # Other Graph Transformations
    def GetSubgraph(self, nodes, keepNodes=True):
        if keepNodes:
            new = {i:[] for i in self._dol.keys()}
        else:
            new = {i:[] for i in nodes}
        for i in nodes:
            new[i] = list(set(self._dol[i]).intersection(set(nodes)))
        return Graph(len(self._dol), new)
    def GetComplement(self):
        new = {i:[] for i in self._dol.keys()}
        for i in self._dol.keys():
            new[i] = list(set(self._dol.keys()) - set(self._dol[i]) - set([i]))
        return Graph(len(new),new)
    def Complement(self): # in-place operator
        for i in self._dol.keys():
            self._dol[i] = list(set(self._dol.keys()) - set(self._dol[i]) - set([i]))
    def Union(self, other):
        if len(self._dol) != len(other._dol):
            raise ValueError("graphs have different number of nodes")
        for i in self._dol.keys():
            self._dol[i] = list(set(self._dol[i]).union(other._dol[i]))
    def BinaryAdd(self, other):
        if len(self._dol) != len(other._dol):
            raise ValueError("graphs have different number of nodes")
        for i in self._dol.keys():
            self._dol[i] = list(set(self._dol[i]).union(other._dol[i]) -
                               set(self._dol[i]).intersection(other._dol[i]))

    # Checking Graph Properties
    def IsEmpty(self):
        for _,l in self._dol.items():
            if l:
                return False
        return True
    def IsClique(self):
        for v in self._dol.keys():
            if set(self._dol[v]) != set(self._dol.keys()) - set([v]):
                return False
        return True
    IsComplete = IsClique

    def IsInCycleSpace(self):
        for _, v in self._dol.items():
            if len(v) % 2 != 0:
                return False
        return True

    # Search Algorithms
    def DFS(self, v, key, visited):
        self.DFSUtil(v, visited)
        if key in visited:
            return True
        else: return False

    def find_one_cycle(self, now, prev, visited):
        for v in self._dol[now]:
            if v == prev: continue
            if v in visited:
                idx = next(i for i in range(len(visited)) if visited[i] == v)
                return visited[idx:]
            else:
                cycle = self.find_one_cycle(v, now, visited + [v])
                if cycle:
                    return cycle
        else:
            return None

    def IsConnnected(self, v, w):
        return self.DFS(v, w, set())

    def ConnectedTo(self, v):
        visited = set()
        self.DFSUtil(v, visited)
        return visited

    def DFSUtil(self, v, visited):
        visited.add(v) # adding in place
        for nb in self._dol[v]:
            if nb not in visited:
                self.DFSUtil(nb, visited)

    # Decomposition Algorithms
    def ConnectedComps(self):
        comps = []
        accounted = set()
        for v in self._dol.keys():
            if v not in accounted:
                c = self.ConnectedTo(v)
                accounted = accounted.union(c)
                comps.append(c)
        return comps

    def MinimalOrdering(self):
        return LEXM(self._dol)

    def PrimeComps(self):
        return primecomps(self._dol)

    def get_all_graphs(n):
        G_list = []
        m = int(n * (n-1) / 2)
        triu_idx = np.triu_indices(n,1)
        for i in range(np.power(2, m)):
            b = format(i,'0' + str(m) + 'b')
            G_list.append(Graph(n))
            for j in range(len(b)):
                if int(b[j]):
                    G_list[-1].AddEdge(triu_idx[0][j], triu_idx[1][j])
        return G_list
