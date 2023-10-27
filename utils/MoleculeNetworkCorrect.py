#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rdkit
from copy import deepcopy
from rdkit import Chem
from utils.MoleculeConvert import Mol2Graph, Graph2Mol

from rdkit import Chem
from utils.MoleculeConvert import Mol2Graph, Graph2Mol

def H_Delete_Correct(graph):
    H_list = []
    for id in graph.nodes():
        if graph.nodes()[id]['element'] == 'H':
            H_list.append(id)
    graph.remove_nodes_from(H_list)
    return graph

def O_Charge_Correct(graph, Nodeid=None):
    if Nodeid is None:
        for id in graph.nodes():
            try:
                graph.nodes()[id]['charge'] = graph.nodes()[id]['charge']
            except:
                graph.nodes()[id]['charge'] = 0
            if graph.nodes()[id]['element'] == 'O':
                valency_sum = 0
                for i in graph.neighbors(id):
                    valency_sum += graph.edges()[i, id]['order']
                if valency_sum > 2:
                    if valency_sum == 3:
                        if graph.edges()[i, id]['order'] == 1.5:
                            continue
                    graph.nodes()[id]['charge'] = int(valency_sum - 2)
    else:
        valency_sum = 0
        for i in graph.neighbors(Nodeid):
            valency_sum += graph.edges()[i, Nodeid]['order']
        if valency_sum > 2:
            if valency_sum == 3:
                if graph.edges()[i, Nodeid]['order'] == 1.5:
                    return graph
            graph.nodes()[Nodeid]['charge'] = int(valency_sum - 2)
    return graph

def C_Charge_Correct(graph, Nodeid=None):
    if Nodeid is None:
        for id in graph.nodes():
            try:
                graph.nodes()[id]['charge'] = graph.nodes()[id]['charge']
            except:
                graph.nodes()[id]['charge'] = 0
            if graph.nodes()[id]['element'] == 'C':
                valency_sum = 0
                for i in graph.neighbors(id):
                    try:
                        valency_sum += graph.edges()[i, id]['order']
                    except:
                        print('error')
                if valency_sum > 4 and valency_sum != 4.5:
                    graph.nodes()[id]['charge'] = int(valency_sum - 4)
    else:
        valency_sum = 0
        for i in graph.neighbors(Nodeid):
            valency_sum += graph.edges()[i, Nodeid]['order']
        if valency_sum > 4 and valency_sum != 4.5:
            graph.nodes()[Nodeid]['charge'] = int(valency_sum - 4)
    return graph

def N_Charge_Correct(graph, Nodeid=None):
    if Nodeid is None:
        for id in graph.nodes():
            try:
                graph.nodes()[id]['charge'] = graph.nodes()[id]['charge']
            except:
                graph.nodes()[id]['charge'] = 0
            if graph.nodes()[id]['element'] == 'N':
                valency_sum = 0
                valency_list = []
                for i in graph.neighbors(id):
                    valency_sum += graph.edges()[i, id]['order']
                    valency_list.append(graph.edges()[i, id]['order'])
                if valency_sum > 3:
                    graph.nodes()[id]['charge'] = int(valency_sum - 3)
    else:
        valency_sum = 0
        valency_list = []
        for i in graph.neighbors(Nodeid):
            valency_sum += graph.edges()[i, Nodeid]['order']
            valency_list.append(graph.edges()[i, id]['order'])
        if valency_sum > 3:
            graph.nodes()[Nodeid]['charge'] = int(valency_sum - 3)
    return graph


def Edge_order_change(graph):
    for i, j in graph.edges():
        if graph.edges()[i, j]['order'] == 12:
            graph.edges()[i, j]['order'] = 1.5
    return graph

def Edge_order_Correct(graph):
    for i, j in graph.edges():
        if graph.edges()[i, j]['order'] == 1.5:
            graph.edges()[i, j]['order'] = 12
    return graph

def NO2_Charge_Correct(graph, Nodeid=None):
    if Nodeid is None:
        for id in graph.nodes():
            try:
                graph.nodes()[id]['charge'] = graph.nodes()[id]['charge']
            except:
                graph.nodes()[id]['charge'] = 0
            if graph.nodes()[id]['element'] == 'O':
                if graph.degree(id) == 1:
                    neighbor = list(graph.neighbors(id))[0]
                    if graph.nodes()[neighbor]['element'] == 'N' and graph.edges()[neighbor, id]['order'] == 1:
                        graph.nodes()[id]['charge'] = -1
    else:
        if graph.nodes()[Nodeid]['element'] == 'O':
            if graph.degree(Nodeid) == 1:
                neighbor = list(graph.neighbors(Nodeid))[0]
                if graph.nodes()[neighbor]['element'] == 'N' and graph.edges()[neighbor, id]['order'] == 1:
                    graph.nodes()[Nodeid]['charge'] = -1
    return graph

def MolGraphCorrect(graphs, node_indicator):
    id_list = []
    for i, graph in enumerate(graphs):
        graph = H_Delete_Correct(graph)
        graph = C_Charge_Correct(graph)
        graph = N_Charge_Correct(graph)
        graph = O_Charge_Correct(graph)
        graph = NO2_Charge_Correct(graph)
        graph = Edge_order_Correct(graph)
        mol = Graph2Mol(graph)

        if mol == None:
            id_list.append(graph.graph['id'])
            graph.graph['right'] = 0
            for gi in range(len(graph.nodes)):
                graph.nodes[gi]['hcount'] = 0
                graph.nodes[gi]['RadElec'] = 0
        else:
            smiles = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                id_list.append(graph.graph['id'])
                graph.graph['right'] = 0
                for gi in range(len(graph.nodes)):
                    graph.nodes[gi]['hcount'] = 0
                    graph.nodes[gi]['RadElec'] = 0
            else:
                mol = Graph2Mol(graph)
                smiles = Chem.MolToSmiles(mol, allHsExplicit=True)
                mol = Chem.MolFromSmiles(smiles)
                Chem.Kekulize(mol, True)
                if len(graphs) == 4337:
                    graph1 = Mol2Graph(mol, node_indicator, H_flag=1)
                else:
                    graph1 = Mol2Graph(mol, node_indicator, H_flag=0)
                graph1.graph = graph.graph
                graph = deepcopy(graph1)
                graph.graph['right'] = 1
                graphs[i] = graph

    return graphs
