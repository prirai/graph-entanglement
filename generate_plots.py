from graph_tools import *
from scipy.sparse import csgraph
from numpy.linalg import eig
import scipy
import numpy
import sys
import matplotlib.pyplot as plt
from progressbar import progressbar
from numpy.linalg import matrix_rank
n = 9
if len(sys.argv) > 1:
    n = int(sys.argv[1])

adj_mats, g6codes = get_adj_mat(n)

G_s = []
for i in progressbar(range(len(adj_mats))):
    G_s.append(nx.from_dict_of_lists(adj_mat_to_adj_list(adj_mats[i])))

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

for i in Ls:
    w, v = eig(i)
    w = sorted(w)
    eigen2.append(w[1].real)

print('Filtered to', len(G_s_filtered), 'graphs.')
Cg = []
for g in G_s_filtered:
    Cg.append(cheeger(g))
print('At collect combs')
d_vals = np.array([collect_combs(i) for i in parity_mats]).astype('float32')

cols = {3: 'red', 4:'green', 5:'blue', 6:'black', 7:'pink'}
for k in range(3,7):
    plt.scatter([eigen2[i] for i in range(len(eigen2)) if d_vals[i] == k],
                [Cg[i] for i in range(len(Cg)) if d_vals[i] == k],    
                color=cols[k], label=k, s=3*k-3)

# plt.scatter([Cg[i] for i in range(len(Cg)) if d_vals[i] == 4], 
#             [eigen2[i] for i in range(len(eigen2)) if d_vals[i] == 4], 
#             color=cols[4], label=k)
plt.ylabel('C(G) values')
plt.xlabel(r'Second smallest eigenvalues $\alpha(G)$')
plt.title(r'C(G) values vs $\alpha(G)$ values for n = {}'.format(n))
plt.legend(title="D values")
plt.savefig(f'out/Cg_eigen2_d_vals_{n}.png', dpi=200)
plt.close()
print(f'Generated for n={n}.')