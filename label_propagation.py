# https://github.com/benedekrozemberczki/LabelPropagation/tree/master
# Benedek Rozemberczki - LabelPropagation

"""Tools to calculate edge scores."""

import networkx as nx
from tqdm import tqdm

def normalized_overlap(g, node_1, node_2):
    """
    Calculating the normalized neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    """
    Calculating the neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    return float(inter)

def unit(g, node_1, node_2):
    """
    Creating unit weights for edge.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    return 1

def min_norm(g, node_1, node_2):
    """
    Calculating the min normalized neighbourhood overlap.
    :param g: NetworkX graph.
    :param node_1: First end node of edge.
    :param node_2: Second end node of edge.
    """
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    return float(inter)/float(min_norm)

def overlap_generator(metric, graph):
    """
    Calculating the overlap for each edge.
    :param metric: Weight metric.
    :param graph: NetworkX object.
    :return : Edge weight hash table.
    """
    edges =[(edge[0], edge[1]) for edge in nx.edges(graph)]
    edges = edges + [(edge[1], edge[0]) for edge in nx.edges(graph)]
    return {edge: metric(graph, edge[0], edge[1]) for edge in tqdm(edges)}

"""Model class label propagation."""

import random
import networkx as nx
from tqdm import tqdm
#from community import modularity
#from print_and_read import json_dumper
#from calculation_helper import overlap, unit, min_norm, normalized_overlap, overlap_generator

class LabelPropagator:
    """
    Label propagation class.
    """
    def __init__(self, graph, args):
        """
        Setting up the Label Propagator object.
        :param graph: NetworkX object.
        :param args: Arguments object.
        """
        self.args = args
        self.seeding = args['seed'] #args.seed
        self.graph = graph
        self.nodes = [node for node in graph.nodes()]
        self.rounds = args['rounds'] #args.rounds
        self.labels = {node: node for node in self.nodes}
        self.label_count = len(set(self.labels.values()))
        self.flag = True
        #self.weight_setup(args.weighting)
        self.weight_setup(args['weighting'])

    def weight_setup(self, weighting):
        """
        Calculating the edge weights.
        :param weighting: Type of edge weights.
        """
        if weighting == "overlap":
            self.weights = overlap_generator(overlap, self.graph)
        elif weighting == "unit":
            self.weights = overlap_generator(unit, self.graph)
        elif weighting == "min_norm":
            self.weights = overlap_generator(min_norm, self.graph)
        else:
            self.weights = overlap_generator(normalized_overlap, self.graph)

    def make_a_pick(self, source, neighbors):
        """
        Choosing a neighbor from a propagation source node.
        :param source: Source node.
        :param neigbors: Neighboring nodes.
        """
        scores = {}
        for neighbor in neighbors:
            neighbor_label = self.labels[neighbor]
            if neighbor_label in scores.keys():
                scores[neighbor_label] = scores[neighbor_label] + self.weights[(neighbor, source)]
            else:
                scores[neighbor_label] = self.weights[(neighbor, source)]
        top = [key for key, val in scores.items() if val == max(scores.values())]
        return random.sample(top, 1)[0]

    def do_a_propagation(self):
        """
        Doing a propagation round.
        """
        random.seed(self.seeding)
        random.shuffle(self.nodes)
        for node in tqdm(self.nodes):
            neighbors = nx.neighbors(self.graph, node)
            pick = self.make_a_pick(node, neighbors)
            self.labels[node] = pick
        current_label_count = len(set(self.labels.values()))
        if self.label_count == current_label_count:
            self.flag = False
        else:
            self.label_count = current_label_count

    def do_a_series_of_propagations(self):
        """
        Doing propagations until convergence or reaching time budget.
        """
        index = 0
        while index < self.rounds and self.flag:
            index = index + 1
            print("\nLabel propagation round: " + str(index)+".\n")
            self.do_a_propagation()
        print("")
        print("Modularity is: " + str(round(nx.community.modularity(self.labels, self.graph), 3)) + ".\n")
        json_dumper(self.labels, self.args.assignment_output)

"""Parsing the parameters."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it does community detection on the Facebook politicians network.
    The default hyperparameters give a good quality clustering. Default weighting happens by neighborhood overlap.
    """
    parser = argparse.ArgumentParser(description="Run Label Propagation.")

    parser.add_argument("--input",
                        nargs="?",
                        default="./data/politician_edges.csv",
	                help="Input graph path.")

    parser.add_argument("--assignment-output",
                        nargs="?",
                        default="./output/politician.json",
	                help="Assignment path.")

    parser.add_argument("--weighting",
                        nargs="?",
                        default="overlap",
	                help="Overlap weighting.")

    parser.add_argument("--rounds",
                        type=int,
                        default=30,
	                help="Number of iterations. Default is 30.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed. Default is 42.")

    return parser.parse_args()

"""Tools for data reading and writing."""

import json
import pandas as pd
import networkx as nx
from texttable import Texttable

def argument_printer(args):
    """
    Function to print the arguments in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(input_path):
    """
    Function to read graph from input path.
    :param input_path: Graph read into memory.
    :return graph: Networkx graph.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def json_dumper(data, path):
    """
    Function to save a JSON to disk.
    :param data: Dictionary of cluster memberships.
    :param path: Path for dumping the JSON.
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

"""Running label propagation."""

#from model import LabelPropagator
#from param_parser import parameter_parser
#from print_and_read import graph_reader, argument_printer

def create_and_run_model(graph, args):
    """
    Method to run the model.
    :param args: Arguments object.
    """
    #graph = graph_reader(args.input)
    model = LabelPropagator(graph, args)
    return model
    model.do_a_series_of_propagations()

if __name__ == "__main__":
    args = parameter_parser()
    argument_printer(args)
    create_and_run_model(args)

