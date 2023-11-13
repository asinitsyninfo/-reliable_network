#https://pyvis.readthedocs.io/en/latest/tutorial.html#node-properties
#https://github.com/anxiaonong/Maxflow-Algorithms/blob/master/Edmonds-Karp%20Algorithm.py
#Network Graph Capacity Optimization with adding one more powerful Edge
#installation:   pip install pyvis


#https://neerc.ifmo.ru/wiki/index.php?title=%D0%A4%D0%B0%D0%BA%D1%82%D0%BE%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_%D0%B3%D1%80%D0%B0%D1%84%D0%BE%D0%B2#1-.D1.84.D0.B0.D0.BA.D1.82.D0.BE.D1.80.D0.B8.D0.B7.D0.B0.D1.86.D0.B8.D1.8F
#https://vuzlit.com/854193/ostovnyy_graf_minimalnogo_vesa
from pyvis.network import Network
import numpy as np
net = Network(height="500px", width="100%", bgcolor="white", font_color="white")
# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph
 
UnSpanTree=[ [],[] ]
SpanTree=[ [],[] ]
C = np.loadtxt("matrix.csv", delimiter=",")
np.matrix(C)
C=C.astype(int)
dim_c=max(C.shape)
#print(max(C.shape));
#print(C);	
UnSpanTree = [[0] * dim_c for i in range(dim_c)]
SpanTree = [[0] * dim_c for i in range(dim_c)]

# Class to represent a graph
class Graph:
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
 
    # Function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    # A utility function to find set of an element i
    # (truly uses path compression technique)
    def find(self, parent, i):
        if parent[i] != i:
 
            # Reassignment of node's parent
            # to root node as
            # path compression requires
            parent[i] = self.find(parent, parent[i])
        return parent[i]
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
 
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[y] = x
            rank[x] += 1
 
    # The main function to construct MST
    # using Kruskal's algorithm
    def KruskalMST(self):
 
        # This will store the resultant MST
        result = []
 
        # An index variable, used for sorted edges
        i = 0
 
        # An index variable, used for result[]
        e = 0
 
        # Sort all the edges in
        # non-decreasing order of their
        # weight
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        # Number of edges to be taken is less than to V-1
        while e < self.V - 1:
 
            # Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
 
            # If including this edge doesn't
            # cause cycle, then include it in result
            # and increment the index of result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge
 
        minimumCost = 0
        #print("Edges in the constructed MST")
        for u, v, weight in result:
            #print (u,v,weight)
            minimumCost += weight
            #print("%d -- %d == %d" % (u, v, weight))
            SpanTree[u][v]+=weight
        #print("Minimum Spanning Tree", minimumCost)
        #print(result)
#graph vizualization1
def graphviz(C):
	num_ed_st=0
	for i in range(len(C)):	
		net.add_node(i, label=str(i),  color='green',shape='cylinder', borderWidth=1, title='Node:'+str(i))
	for i in range(len(C)):
		for j in range(i,len(C)):
			if(C[i][j]!=0):
				color="green"
				label=str(int(C[i][j]))
				txt=''
				if(SpanTree[i][j]!=0):
					color = 'red'
					label=str(int(SpanTree[i][j]))
					txt=' SpTr:' 
					num_ed_st+=1 		
				net.add_edge(i, j,label=txt+label, color=color, title=txt+'('+str(i)+','+str(j)+')'+':'+label+'/'+str(int(C[i][j])))
	net.toggle_physics(True)
	net.show("KruskalOpt.html")
	print('num_ed_st=',num_ed_st)
	return None 
 
# Driver code
def CalcSpanTree(C):
	if __name__ == '__main__':
    		g = Graph(dim_c)
    		for i in range(len(C)):
      			for j in range(i,len(C)):
      				if ( C[i][j] !=0) :
       					g.addEdge(i, j, C[i][j])
# Function call
    		g.KruskalMST()
	return None	
#Main
print(C)
for i in range(len(C)):
	for j in range(i,len(C)):
		if ( C[i][j] !=0) :    
		    dummy=C[i][j]
		    C[i][j]=0
		    print('\nExcluded C(',i,',',j,')\n',C,'\n')			
		    CalcSpanTree(C)
		    print("\nSpanTree= \n",np.array(SpanTree),'\n')
		    C[i][j]=dummy
graphviz(C)
# This code is contributed by Neelam Yadav
# Improved by James Graça-Jones
