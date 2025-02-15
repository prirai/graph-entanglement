from graph_tools import *
from scipy.sparse import csgraph
from numpy.linalg import eig
import matplotlib.pyplot as plt
import scipy
import sys
import time


for n in range(4,10):
    adj_mats, g6codes = get_adj_mat(n)
    G_s = [nx.from_dict_of_lists(adj_mat_to_adj_list(i)) for i in adj_mats]

    G_s_filtered = []
    Ls = []
    eigen2 = []
    # parity_mats = [get_parity_matrix(n, graph).astype(np.int_) for graph in G_s]
    parity_mats = []
    for i, graph in enumerate(G_s):
        if i % 500 == 0:
            print(i, 'done!')
        p = get_parity_matrix(n, graph)
        if p is not None:
            parity_mats.append(p.astype(np.int_))
            G_s_filtered.append(G_s[i])
            Ls.append(csgraph.laplacian(nx.adjacency_matrix(graph).todense()))

    for i in parity_mats:
        col = i.shape[1]
        row = i.shape[0]
        # print(f'({col}, {row})')

    for i in Ls:
        w, v = eig(i)
        w = sorted(w)
        eigen2.append(w[1].real)

    # print(eigen2)
    # if len(G_s_filtered) == len(parity_mats):
    #     print(True)

    Cg = []
    for g in G_s_filtered:
        Cg.append(cheeger(g))

    # print(Cg)
    print('At collect combs')
    d_vals = np.array([collect_combs(i) for i in parity_mats]).astype('float32')
    print(d_vals)
    # print(Cg)
    res = [n, scipy.stats.pearsonr(Cg, d_vals).pvalue, scipy.stats.pearsonr(Cg, eigen2).pvalue, scipy.stats.pearsonr(eigen2, d_vals).pvalue]
    print(res)
    np.savetxt(f'out/{n}_out.txt', res)
    # plt.scatter(Cg, d_vals)
    # plt.savefig(f'out/{n}_Cg_d.png', dpi=200)
    # plt.close()
    # for i, e in enumerate(G_s_filtered):
    #     nx.draw(e)
    #     plt.savefig(f'imgs/{n}_{str(i).zfill(3)}.png', dpi=50)
    #     plt.close()
