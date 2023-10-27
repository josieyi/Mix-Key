
import copy
import time
import rdkit
import random
import torch
from configs import arg_parse_clf
import numpy as np
import networkx as nx
from rdkit import Chem
from utils.MoleculeConvert import Graph2Mol, Extract_Scaffold
from rdkit.Chem.Scaffolds import MurckoScaffold

args = arg_parse_clf()
if (torch.cuda.is_available() and args.cuda):
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# scaffold model
def scaffold_model(G0:nx.Graph, node_labels, rate=0.1):
    if G0.graph['right'] == 0:
        return G0
    if not nx.is_connected(G0):
        sub_graphs = list(G0.subgraph(c) for c in nx.connected_components(G0))
        for i, g in enumerate(sub_graphs):
            if g.number_of_edges() < 4:
                if i == 0:
                    G_new = copy.deepcopy(g)
                    continue
                else:
                    G_s = copy.deepcopy(g)
                    G_new = nx.compose(G_new, G_s)
            else:
                g_old = copy.deepcopy(g)
                g_new = line(g, node_labels, rate)
                if g_old.number_of_nodes() == g_new.number_of_nodes():
                    if i == 0:
                        G_new = copy.deepcopy(g_new)
                        continue
                    else:
                        G_s = copy.deepcopy(g_new)
                        G_new = nx.compose(G_new, G_s)
                else:
                    if i == 0:
                        G_new = copy.deepcopy(g_old)
                        continue
                    else:
                        G_s = copy.deepcopy(g_old)
                        G_new = nx.compose(G_new, G_s)

        if G0.number_of_nodes() == G_new.number_of_nodes():
            G_new.graph['stand'] = 1
            return G_new
        else:
            G0.graph['stand'] = 0
            return G0
    else:
        G_new = line(G0, node_labels, rate)
        G_new.graph['stand'] = 1
        return G_new


def line(G0: nx.Graph, node_labels, rate=0.1):
    try:
        scaffold0, _ = Extract_Scaffold(G0, node_labels)
    except:
        return G0
    linegraph0 = nx.line_graph(scaffold0)
    edge = list(linegraph0.nodes())
    nswap = rate * len(edge)

    if len(edge) != 0:
        G1 = copy.deepcopy(G0)
        G2 = copy.deepcopy(G0)
        G2.graph['stand'] = 0
        swapcount = 0
        n = 0
        while swapcount < nswap and n < 100:
            try:
                scaffold, _ = Extract_Scaffold(G1, node_labels)
            except:
                G1.graph['stand'] = 0
                n = n + 1
                continue
            linegraph = nx.line_graph(scaffold)
            edge = list(linegraph.nodes())
            node = list(linegraph.nodes())
            edge_ty = nx.get_edge_attributes(scaffold, name="order")
            edge_label = [edge_ty.get(e) for e in edge]
            node_ori = [scaffold.nodes()[s]['ori_pos'] for s in range(scaffold.number_of_nodes())]
            a_l = torch.FloatTensor(nx.to_numpy_matrix(linegraph)).to(device)

            a = random.sample(edge, 1)
            u = node_ori[a[0][0]]
            v = node_ori[a[0][1]]
            q = edge_label[edge.index(a[0])]
            A = [a_l[edge.index(a[0])] * a_l[i] for i in range(len(a_l))]
            nodeselect = copy.deepcopy(node)
            for si, s in enumerate(A):
                if edge_label[si] != q:
                    nodeselect.remove(node[si])
                    continue
                if a_l[si][edge.index(a[0])] != 0:
                    nodeselect.remove(node[si])
                    continue
                if torch.count_nonzero(s).item() == 0:
                    ind = [idd.item() for idd in torch.nonzero(s)]
                    for i in ind:
                        if torch.count_nonzero(a_l[edge.index(a[0])] * a_l[i]).item() != 0 or torch.count_nonzero(
                                a_l[si] * a_l[i]).item() != 0:
                            try:
                                nodeselect.remove(node[si])
                            except:
                                pass
            try:
                b = random.sample(nodeselect, 1)
                x = node_ori[b[0][0]]
                y = node_ori[b[0][1]]
                try:
                    G1.remove_edge(x, y)
                except:
                    G1.graph['stand'] = 0
                    G1 = copy.deepcopy(G2)
                    n = n + 1
                    continue
                try:
                    G1.remove_edge(u, v)
                except:
                    G1.graph['stand'] = 0
                    G1 = copy.deepcopy(G2)
                    n = n + 1
                    continue
                if q == 0:
                    q = 1
                try:
                    if ((u, y) not in G1.edges) and ((x, v) not in G1.edges):
                        G1.add_edge(u, y)
                        G1.add_edge(x, v)
                        G1.add_edges_from([(u, y, {'order': q})])
                        G1.add_edges_from([(x, v, {'order': q})])
                        swapcount += 1
                    else:
                        G1.graph['stand'] = 0
                        G1 = copy.deepcopy(G2)
                        n = n + 1
                        continue
                except:
                    if ((u, x) not in G1.edges) and ((y, v) not in G1.edges):
                        G1.add_edge(u, x)
                        G1.add_edge(y, v)
                        G1.add_edges_from([(u, x, {'order': q})])
                        G1.add_edges_from([(y, v, {'order': q})])
                        swapcount += 1
                    else:
                        G1.graph['stand'] = 0
                        G1 = copy.deepcopy(G2)
                        n = n + 1
                        continue
            except:
                G1.graph['stand'] = 0
                n = n + 1
                continue
            if not nx.is_connected(G1):
                G1.graph['stand'] = 0
                G1 = copy.deepcopy(G2)
                swapcount -= 1
                n = n + 1
                continue

            try:
                G1 = check_graph(G1)
            except:
                G1 = copy.deepcopy(G2)
                swapcount -= 1
                G1.graph['stand'] = 0
                continue
            if G1.graph['stand'] == 0:
                G1 = copy.deepcopy(G2)
                swapcount -= 1
                continue
            else:
                G2 = copy.deepcopy(G1)
                G2.graph['stand'] = 1
                break
            n = n + 1
        if G2.graph['stand'] == 0:
            G0.graph['stand'] = 0
            return G0
        else:
            try:
                G2 = check_graph(G2)
                return G2
            except:
                G0.graph['stand'] = 0
                return G0
    else:
        G0.graph['stand'] = 0
        return G0

# group model
def group_model(G0:nx.Graph, node_labels, rate=0.1):
    if G0.graph['right'] == 0:
        return G0
    if not nx.is_connected(G0):
        sub_graphs = list(G0.subgraph(c) for c in nx.connected_components(G0))
        for i, g in enumerate(sub_graphs):
            if g.number_of_edges() < 4:
                if i == 0:
                    G_new = copy.deepcopy(g)
                    continue
                else:
                    G_s = copy.deepcopy(g)
                    G_new = nx.compose(G_new, G_s)
            else:
                g_old = copy.deepcopy(g)
                g_new = group(g, rate)
                if g_old.number_of_nodes() == g_new.number_of_nodes():
                    if i == 0:
                        G_new = copy.deepcopy(g_new)
                        continue
                    else:
                        G_s = copy.deepcopy(g_new)
                        G_new = nx.compose(G_new, G_s)
                else:
                    if i == 0:
                        G_new = copy.deepcopy(g_old)
                        continue
                    else:
                        G_s = copy.deepcopy(g_old)
                        G_new = nx.compose(G_new, G_s)

        if G0.number_of_nodes() == G_new.number_of_nodes():
            G_new.graph['stand'] = 1
            return G_new
        else:
            G0.graph['stand'] = 0
            return G0
    else:
        G_new = group(G0, rate)
        G_new.graph['stand'] = 1
        return G_new

def group(G0:nx.Graph, rate=0.1):
    swapcount = 0
    n1 = 0
    edge_num = G0.number_of_edges()
    nswap = rate * edge_num

    G = copy.deepcopy(G0)
    G2 = copy.deepcopy(G0)
    G2 = nx.Graph(G2)
    G2.graph['stand'] = 0

    mol = Graph2Mol(G0)
    all_node_idx = []
    for i in mol.GetAtoms():
        i.SetIntProp("atom_idx", i.GetIdx())
        all_node_idx.append(i.GetIdx() + min(list(G0.nodes)))
    mol_s = MurckoScaffold.GetScaffoldForMol(mol)
    s_node_idx = []
    for i in mol_s.GetAtoms():
        idx = i.GetIntProp("atom_idx")
        s_node_idx.append(idx + min(list(G0.nodes)))
    g_node_idx = copy.deepcopy(all_node_idx)
    for a in g_node_idx[:]:
        if a in s_node_idx:
            g_node_idx.remove(a)

    candidate_node = copy.deepcopy(g_node_idx)
    candidate_edge = list(G.edges(candidate_node))

    if len(candidate_edge) == 0 or len(candidate_node) < 2:
        G0.graph['stand'] = 0
        return G0

    while swapcount < nswap and n1 < (len(candidate_edge) * len(candidate_edge)):
        candidate_edge = list(G2.edges(candidate_node))
        if len(candidate_edge) == 0:
            G2 = copy.deepcopy(G)
            G2.graph['stand'] = 0
            return G2
        del_edge = random.sample(candidate_edge, 1)
        q = G2.edges[del_edge[0]]['order']
        u = del_edge[0][0]
        v = del_edge[0][1]
        G2 = nx.Graph(G2)
        G2.remove_edge(u, v)
        G2.graph['stand'] = 0
        G2.nodes()[u]['hcount'] = G2.nodes()[u]['hcount'] + q
        G2.nodes()[v]['hcount'] = G2.nodes()[v]['hcount'] + q
        G1 = copy.deepcopy(G2)
        n = 0
        while swapcount < nswap and n < 100:
            del_nodes = random.sample(candidate_node, 2)
            x = del_nodes[0]
            y = del_nodes[1]
            if ((x, y) not in G.edges()) and ((x, y) not in G2.edges()) and ((x, y) not in G0.edges()):
                G2 = nx.Graph(G2)
                G2.add_edge(x, y)
                G2.add_edges_from([(x, y, {'order': q})])
            else:
                G2.graph['stand'] = 0
                n = n + 1
                continue
            if not nx.is_connected(G2):
                G2.graph['stand'] = 0
                G2.remove_edge(x, y)
                n = n + 1
                continue
            else:
                G2.nodes()[x]['hcount'] = G2.nodes()[x]['hcount'] - q
                G2.nodes()[y]['hcount'] = G2.nodes()[y]['hcount'] - q
                if G2.nodes()[x]['hcount'] < 0:
                    G2.nodes()[x]['charge'] = G2.nodes()[x]['charge'] + G2.nodes()[x]['hcount']
                    G2.nodes()[x]['hcount'] = 0
                if G2.nodes()[y]['hcount'] < 0:
                    G2.nodes()[y]['charge'] = G2.nodes()[y]['charge'] + G2.nodes()[y]['hcount']
                    G2.nodes()[y]['hcount'] = 0
                G2 = check_graph(G2)
            if G2.graph['stand'] == 1:
                G = copy.deepcopy(G2)
                break
            else:
                G2.graph['stand'] = 0
                G2 = copy.deepcopy(G1)
                n = n + 1
        if G2.graph['stand'] == 1:
            swapcount = swapcount + 1
        else:
            G2 = copy.deepcopy(G)
            n1 = n1 + 1

    if G2.graph['stand'] == 0:
        G0.graph['stand'] = 0
        return G0
    else:
        try:
            G2 = check_graph(G2)
            return G2
        except:
            G0.graph['stand'] = 0
            return G0

def check_graph_rules_strict(Graph):
    mol = Graph2Mol(Graph)
    if mol is None:
        Graph.graph['stand'] = 0
    try:
        mol = Chem.MolToSmiles(mol)
        if mol == 'Se':
            mol = '[Se]'
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            Graph.graph['stand'] = 0
        else:
            Graph.graph['stand'] = 1
    except:
        Graph.graph['stand'] = 0
    return Graph


def check_graph(Graph):
    rtr = check_graph_rules_strict(Graph)
    return rtr


