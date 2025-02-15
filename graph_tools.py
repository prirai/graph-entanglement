import os
import numpy as np
import networkx as nx
import itertools
import subprocess

def get_adj_mat(n):
    file_dir = "./graph_data/"
    file_name = f"graph{n}c.g6"
    f_path = file_dir + file_name

    output = subprocess.check_output(["showg", "-A", f_path], universal_newlines=True)

    adjmats = []
    g6codes = []

    text_to_list = lambda text: [list(map(int, row.split())) for row in text.split("\n")]

    graphs = output.split("\n\n")

    adjmats = [None] * len(graphs)
    g6codes = [None] * len(graphs)

    for i, g in enumerate(graphs):
        op = g.strip("\n").split(".\n")[1]
        adjmats[i] = np.array(text_to_list(op))
        with open(f_path, "r") as file_contents:
            lines = file_contents.readlines()
            g6codes[i] = lines[i]

    return adjmats, g6codes


def adj_mat_to_adj_list1(am):
    al = {}
    for x, row in enumerate(am):
        al[x] = []
        for i, v in enumerate(row):
            if v == 1 and i != x:
                al[x].append(i)
    return al

def adj_mat_to_adj_list(am):
    al = {}
    n = len(am)
    for x in range(n):
        neighbors = [i for i, v in enumerate(am[x]) if v == 1 and i != x]
        if neighbors:
            al[x] = neighbors
    return al


def find_d(matrix): 
    n = matrix.shape[0]
    for i in range(n):
        pivot_row = None
        for j in range(i, n):
            if matrix[j, i] == 1:
                pivot_row = j
                break
        if pivot_row is None:
            continue 
        if pivot_row != i:
            temp = matrix[i, :].copy()
            matrix[i, :] = matrix[pivot_row, :]
            matrix[pivot_row, :] = temp
        for j in range(n):
            if j != i and matrix[j, i] == 1:
                matrix[j, :] = (matrix[j, :] + matrix[i, :]) % 2
    
    rank = np.sum(np.any(matrix, axis=1))
    return rank
    
    
from itertools import combinations

def column_combinations1(matrix):
    n_cols = matrix.shape[1]
    col_indices = range(n_cols)
    col_combinations = {}
    
    for r in range(2, n_cols + 1):
        col_combinations[r] = [matrix[:, cols] for cols in combinations(col_indices, r)]
    return col_combinations

def collect_combs1(matrix):
    n = matrix.shape[0]
    col_combs = column_combinations1(matrix)
    for num_cols, combs in col_combs.items():
        for comb in combs:
            res = np.sum(comb, axis = 1)
            if not np.any(np.mod(res, 2)):
                return num_cols
    print(matrix)
    return n

def collect_combs(matrix):
    n_rows, n_cols = matrix.shape
    
    for r in range(2, n_cols + 1):
        col_indices = combinations(range(n_cols), r)
        for cols in col_indices:
            res = np.sum(matrix[:, cols], axis=1)
            if not np.any(np.mod(res, 2)):
                return r
    
    return n_rows

            
# Not used yet?
def make_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    return gr

def get_common_edges(test, base):
    """Two different lists are passed, one is a typical 
    graph while the other is a complete graph of the same
    n. These lists are obtained using G.edges."""
    l = []
    for i, e in enumerate(base):
        if e in test:
            l.append(i)
    return l

# WIP
def get_parity_matrix1(n, G):
    G0 = nx.complete_graph(n)
    base_edges = list(G0.edges)
    common_edge_numbers = get_common_edges(G.edges, G0.edges)
    mat = np.zeros((n, n*(n-1)//2), dtype=np.bool_)
    for i, e in enumerate(common_edge_numbers):
        ed_in = base_edges[e]
        mat[ed_in[0]][e] = mat[ed_in[1]][e] = 1
    mat1 = mat[:,~np.all(mat == 0, axis = 0)]
    if mat1.shape[1] > mat1.shape[0]:
        return mat1
    else:
        return None
    
def get_parity_matrix(n, G):
    G_edges_set = set(G.edges)
    mat = np.zeros((n, len(G_edges_set)), dtype=np.bool_)
    for i, e in enumerate(G_edges_set):
        mat[e[0], i] = mat[e[1], i] = 1
    
    mat1 = mat[:, ~np.all(mat == 0, axis=0)]
    
    if mat1.shape[1] > mat1.shape[0]:
        return mat1
    else:
        return None
    
def cheeger(G_1):
    ls_cv = []
    n = nx.number_of_nodes(G_1)
    for i in range(2, n//2 + 1):
        for nodes in itertools.combinations(G_1, i):
            edge_set = set()
            for j in nodes:
                edges_j = set(map(lambda t: tuple(sorted(t)), G_1.edges(j)))
                edge_set.update(edges_j)
            num_dels = 0
            num_s = 0
            for e in edge_set:
                if e[0] in nodes and e[1] in nodes:
                    num_s += 1
                else:
                    num_dels += 1
            if num_s != 0: 
                ls_cv.append(num_dels/num_s)
            else:
                continue
    return min(ls_cv)

