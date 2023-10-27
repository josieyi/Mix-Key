#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import rdkit
import logging
import numpy as np
import networkx as nx
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import RDConfig

RDLogger.DisableLog('rdApp.*')


def Extract_Scaffold(Graph, node_labels, verbose=True):
    img = None
    mol = Graph2Mol(Graph)
    for a in mol.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
    mol1 = copy.deepcopy(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None:
            Chem.Kekulize(scaffold, True)

        graph = Mol2Graph(scaffold, node_labels, H_flag=0)
        graph.graph['id'] = Graph.graph['id']
        graph.graph['stand'] = Graph.graph['stand']
        graph.graph['scaffold'] = 1
        graph.graph['label'] = Graph.graph['label']
        if len(graph) == 0:
            scaffold = mol1
            graph = Mol2Graph(scaffold, node_labels, H_flag=0)
            graph.graph['scaffold'] = 0
            graph.graph['id'] = Graph.graph['id']
            graph.graph['stand'] = Graph.graph['stand']
            graph.graph['label'] = Graph.graph['label']
            if scaffold is not None:
                Chem.Kekulize(scaffold, True)
        if verbose:
            img = Draw.MolToImage(scaffold)
    except:
        scaffold = mol1
        graph = Mol2Graph(scaffold, node_labels, H_flag=0)
        graph.graph['scaffold'] = 0
        graph.graph['id'] = Graph.graph['id']
        graph.graph['stand'] = Graph.graph['stand']
        graph.graph['label'] = Graph.graph['label']
    return graph, img

def Sanitize_smiles(smiles):
    '''

    :param smiles:
    :return: standardized smiles
    '''
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
    if 'se' in smiles:
        smiles = smiles.replace('se', 'Se')
    return smiles

def Mol2Graph(mol, node_labels, H_flag):
    '''

    :param mol: molecule
    :param node_labels: node label map
    :return: graph
    :des: convert atoms to nodes; convert bonds to edges
    '''
    graph = nx.Graph()
    if H_flag == 1:
        mol = Chem.AddHs(mol)
    # node_label-element map
    node_labels = dict(zip(node_labels.values(), node_labels.keys()))
    # edge_order-bond map
    bond_labels = dict(zip(rdkit.Chem.rdchem.BondType.values.values(), rdkit.Chem.rdchem.BondType.values.keys()))
    bonds = mol.GetBonds()
    # Get edge list with orders
    bond_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'order': int(bond.GetBondType())}) for bond in bonds]
    for id, node in enumerate(mol.GetAtoms()):
        graph.add_node(id)
        if node.HasProp("__origIdx"):
            graph.nodes[id]['ori_pos'] = node.GetIntProp("__origIdx")
        graph.nodes[id]['element'] = mol.GetAtoms()[id].GetSymbol()
        graph.nodes[id]['charge'] = mol.GetAtoms()[id].GetFormalCharge()
        graph.nodes[id]['hcount'] = mol.GetAtoms()[id].GetTotalNumHs()
        graph.nodes[id]['aromatic'] = mol.GetAtoms()[id].GetIsAromatic()
        graph.nodes[id]['RadElec'] = mol.GetAtoms()[id].GetNumRadicalElectrons()
        node_label_one_hot = [0] * len(node_labels)
        node_label = int(node_labels.get(graph.nodes[id]['element']))
        node_label_one_hot[node_label] = 1
        graph.nodes[id]['label'] = node_label_one_hot
    graph.add_edges_from(bond_list)

    return graph

def Graph2Mol(graph_o):
    '''

    :param graph: molecule graph
    :return: molecule
    '''
    flag = 0
    # Get atoms and adjacency
    graph = copy.deepcopy(graph_o)
    new_node_labels = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, new_node_labels)
    node_list = list(nx.get_node_attributes(graph, 'element').values())
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    bonds = nx.get_edge_attributes(graph, 'order')
    for i, edge in enumerate(graph.edges()):
        adjacency_matrix[edge] = bonds[edge]
        adjacency_matrix[edge[1], edge[0]] = bonds[edge]
    adjacency_matrix = np.array(adjacency_matrix)
    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for id, node in enumerate(graph.nodes):
        a = Chem.Atom(node_list[id])
        molIdx = mol.AddAtom(a)
        try:
            mol.GetAtoms()[id].SetFormalCharge(graph.nodes[id]['charge'])
        except:
            flag = 1
        try:
            mol.GetAtoms()[id].SetIsAromatic(graph.nodes[id]['aromatic'])
        except:
            flag = 2
        try:
            hcount = graph.nodes[id]['hcount']
            mol.GetAtoms()[id].SetNumExplicitHs(hcount)
        except:
            flag = 3
        try:
            RadElec = graph.nodes[id]['RadElec']
            mol.GetAtoms()[id].SetNumRadicalElectrons(RadElec)
        except:
            flag = 4
        node_to_idx[id] = molIdx
    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue
            if bond == 0:
                continue
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], rdkit.Chem.rdchem.BondType.values[bond])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol, True)
    return mol

class MoleculeGraph():
    '''
        1、Mol-Graph转换
        2、官能团采样
        3、骨架提取
        4、N原子修复
        5、合理数据筛选
    '''
    def __init__(self, Graphs, node_labels, Groups=None):

        self.node_labels = node_labels
        self.mols, self.graphs, self.smiles = self.GraphFilter(Graphs)
        self.graph_ids = [graph.graph['id'] for graph in Graphs]

        if Groups is not None:
            Group_ids = self.Sample_Groups()
            if 27 in Group_ids:
                element = list(self.node_labels.values())
                if 'Br' in element:
                    Group_ids.append(39)
                if 'Cl' in element:
                    Group_ids.append(40)
                if 'F' in element:
                    Group_ids.append(41)
                if 'I' in element:
                    Group_ids.append(42)
                if 'At' in element:
                    Group_ids.append(43)
                if 'Ts' in element:
                    Group_ids.append(44)
                Group_ids.remove(27)
            self.groups = self.Group_label_set(list(np.array(Groups)[Group_ids]))

    def Delete_Atom(self, Graph, atom):

        graph = copy.deepcopy(Graph)
        for id, node in enumerate(graph.nodes):
            if graph.nodes[id]['element'] == atom:
                graph.remove_node(id)
        return graph

    def GraphFilter(self, Graphs):
        '''
        滤掉化学键有问题的分子
        :param Graphs:
        :return: Right molecules
        '''
        # Er_mol = []
        mols = []
        smiles = []
        for graph in Graphs:
            mol = Graph2Mol(graph)
            mol = Chem.MolToSmiles(mol, allHsExplicit=True)
            mols.append(Chem.MolFromSmiles(mol))
            smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(mol), allBondsExplicit=True))
        return mols, Graphs, smiles

    def Sample_Groups(self):

        fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
        fparams = FragmentCatalog.FragCatParams(1, 6, fName)
        Groupids = []
        for j, mol in enumerate(self.mols):
            fcat = FragmentCatalog.FragCatalog(fparams)
            fcgen = FragmentCatalog.FragCatGenerator()
            fcgen.AddFragsFromMol(mol, fcat)
            num_entries = fcat.GetNumEntries()
            temp = []
            for i in range(0, num_entries):
                temp.extend(list(fcat.GetEntryFuncGroupIds(i)))
            Groupids.extend(set(temp))
        Groupids = list(set(Groupids))
        Groupids.sort()
        return Groupids

    def Extract_Scaffold(self, Graph, verbose=True):

        img = None
        mol = Graph2Mol(Graph)
        for a in mol.GetAtoms():
            a.SetIntProp("__origIdx", a.GetIdx())
        mol1 = copy.deepcopy(mol)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None:
                Chem.Kekulize(scaffold, True)
            graph = Mol2Graph(scaffold, self.node_labels, H_flag=0)
            graph.graph['id'] = Graph.graph['id']
            graph.graph['stand'] = Graph.graph['stand']
            graph.graph['scaffold'] = 1
            graph.graph['label'] = Graph.graph['label']
            if len(graph) == 0:
                scaffold = mol1
                graph = Mol2Graph(scaffold, self.node_labels, H_flag=0)
                graph.graph['scaffold'] = 0
                graph.graph['id'] = Graph.graph['id']
                graph.graph['stand'] = Graph.graph['stand']
                graph.graph['label'] = Graph.graph['label']
                if scaffold is not None:
                    Chem.Kekulize(scaffold, True)
            if verbose:
                img = Draw.MolToImage(scaffold)
        except:
            scaffold = mol1
            graph = Mol2Graph(scaffold, self.node_labels, H_flag=0)
            graph.graph['scaffold'] = 0
            graph.graph['id'] = Graph.graph['id']
            graph.graph['stand'] = Graph.graph['stand']
            graph.graph['label'] = Graph.graph['label']
        return graph, img

    def Symbol_fix(self, smiles):
        for atom in self.node_labels.values():
            if len(atom) > 1:
                temp = copy.copy(atom)
                if temp.lower() in smiles:
                    smiles = smiles.replace(atom.lower(), atom)
        return smiles

    def Group_label_set(self, graphs):

        node_labels = dict(zip(self.node_labels.values(), self.node_labels.keys()))
        for i, graph in enumerate(graphs):
            graph.graph['id'] = i
            for id, node in enumerate(graph.nodes):
                node_label_one_hot = [0] * len(self.node_labels)
                node_label = node_labels.get(graph.nodes[id]['element'])
                node_label_one_hot[node_label] = 1
                graph.nodes[id]['label'] = node_label_one_hot
        return graphs