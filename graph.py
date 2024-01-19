from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from collections import defaultdict
from pathlib import Path 
import json

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
import pygraphviz as pgv

from visualization import EDGE_TYPE_COLORS, generate_random_color

class Node:
    """
    A node in our computational graph. If pos is None, positions are ignored. The in_hook is the TL hook into its inputs, 
    while the out_hook gets its outputs. The index indexes into position (and also heads)
    """
    name: str
    layer: int
    pos: Optional[int]
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer:int, pos:Optional[int], in_hook: List[str], out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.pos = slice(None) if pos is None else pos
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Node({self.name})'
    
    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int, pos: Optional[int]=None):
        name = 'logits' if pos is None else f'logits_{pos}'
        index = slice(None) if pos is None else (slice(None), pos)
        super().__init__(name, n_layers - 1, pos, f"blocks.{n_layers - 1}.hook_resid_post", '', index)
        
class MLPNode(Node):
    def __init__(self, layer: int, pos: Optional[int]=None):
        name = f'm{layer}' if pos is None else f'm{layer}_{pos}'
        index = slice(None) if pos is None else (slice(None), pos)
        super().__init__(name, layer, pos, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int, pos: Optional[int]=None):
        name = f'a{layer}.h{head}' if pos is None else f'a{layer}.h{head}_{pos}'
        self.head = head
        index = (slice(None), slice(None), head) if pos is None else (slice(None), pos, head)
        super().__init__(name, layer, pos, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self, pos:Optional[int]=None):
        name = 'input' if pos is None else f'input_{pos}'
        index = slice(None) if pos is None else (slice(None), pos)
        super().__init__(name, 0, pos, '', "blocks.0.hook_resid_pre", index)

class Edge:
    name: str
    parent: Node 
    child: Node 
    hook: str
    index: Tuple
    score : Optional[float]
    in_graph: bool
    def __init__(self, parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent 
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            #if parent.pos is not None:
            self.index = (slice(None), parent.pos, child.head)
            #else:
            #    self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook
    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF00FF"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Edge({self.name})'
    
    def __hash__(self):
        return hash(self.name)

class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    logits: List[Node]
    n_pos: Optional[int]
    cfg: HookedTransformerConfig

    def __init__(self, nodes: List[Node], edges:Dict[Node, Edge], logits=List[Node]):
        self.nodes = nodes
        self.edges = edges
        self.logits = logits

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def scores(self, nonzero=True, in_graph=False):
        return torch.tensor([edge.score for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)]) if nonzero else torch.tensor([edge.score for edge in self.edges.values()])

    def parent_node_names(self):
        return {edge.parent.out_hook for edge in self.edges.values()}
    
    def child_node_names(self):
        return {edge.hook for edge in self.edges.values()}

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())
    
    def apply_threshold(self, threshold: float, absolute: bool):
        for edge in self.edges.values():
            edge.in_graph = abs(edge.score) >= threshold if absolute else edge.score >= threshold

    def prune_dead_nodes(self, prune_childless_attn=False, prune_childless=False, prune_parentless=False):
        for logit in self.logits:
            logit.in_graph = any(parent_edge.in_graph for parent_edge in logit.parent_edges)

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue 
            
            if any(child_edge.in_graph for child_edge in node.child_edges) :
                node.in_graph = True
            else:
                if prune_childless or (prune_childless_attn and isinstance(node, AttentionNode)):
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else: 
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True 
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False 
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False


    @classmethod
    def from_model(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict]):
        graph = Graph({}, {}, [])
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        else:
            graph.cfg = model_or_config
        
        graph.n_pos = None
        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            # at last layer, make sure only the lastmost components occur (maybe not necessary)
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer)
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node 
            graph.nodes[mlp_node.name] = mlp_node     
                                    
            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:          
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)
                
                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:     
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)
                        
        logit_node = LogitNode(graph.cfg['n_layers'])
        for node in residual_stream:
            graph.add_edge(node, logit_node)
            
        graph.logits = [logit_node] 
        graph.nodes[logit_node.name] = logit_node

        return graph


    @classmethod
    def from_model_positional(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict], input_length:int):
        graph = Graph({}, {}, [])
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        else:
            graph.cfg = model_or_config

        graph.n_pos = input_length
        residual_stream_by_pos = []
        for pos in range(graph.n_pos):
            input_node = InputNode(pos)
            graph.nodes[input_node.name] = input_node 
            residual_stream_by_pos.append([input_node])

        for layer in range(graph.cfg['n_layers']):
            # first generate and add nodes in the right order   
            attn_nodes_by_pos = [[AttentionNode(layer, head, pos) for head in range(graph.cfg['n_heads'])] for pos in range(graph.n_pos)]  
            mlp_node_by_pos = [MLPNode(layer, pos) for pos in range(graph.n_pos)]     
            for attn_nodes, mlp_node in zip(attn_nodes_by_pos, mlp_node_by_pos):
                for attn_node in attn_nodes:
                    graph.nodes[attn_node.name] = attn_node
                graph.nodes[mlp_node.name] = mlp_node

            # then add the edges (reverse)
            for pos in range(graph.n_pos -1 , -1, -1):
                attn_nodes = attn_nodes_by_pos[pos]
                mlp_node = mlp_node_by_pos[pos]

                if graph.cfg['parallel_attn_mlp']:
                    for node in residual_stream_by_pos[pos]:
                        graph.add_edge(node, mlp_node)
                        for attn_node in attn_nodes:
                            graph.add_edge(node, attn_node, qkv='q')
                    for residual_stream in residual_stream_by_pos[:pos+1]:
                        for node in residual_stream:
                            for attn_node in attn_nodes:                
                                for letter in 'kv':           
                                    graph.add_edge(node, attn_node, qkv=letter)
                    
                    residual_stream_by_pos[pos] += attn_nodes
                    residual_stream_by_pos[pos].append(mlp_node)

                else:
                    for node in residual_stream_by_pos[pos]:
                        for attn_node in attn_nodes:
                                graph.add_edge(node, attn_node, qkv='q')

                    for residual_stream in residual_stream_by_pos[:pos+1]:
                        for node in residual_stream:
                            for attn_node in attn_nodes:                
                                for letter in 'kv':           
                                    graph.add_edge(node, attn_node, qkv=letter)
                    residual_stream_by_pos[pos] += attn_nodes

                    for node in residual_stream_by_pos[pos]:
                        graph.add_edge(node, mlp_node)
                    residual_stream_by_pos[pos].append(mlp_node)

        logits = []
        for pos in range(graph.n_pos):
            logit_node = LogitNode(graph.cfg['n_layers'], pos)
            graph.nodes[logit_node.name] = logit_node
            logits.append(logit_node)
            for node in residual_stream_by_pos[pos]:
                graph.add_edge(node, logit_node)
            
        graph.logits = logits 

        return graph
    
    def to_json(self, filename):
        # non serializable info
        d = {'cfg':self.cfg, 'nodes': {name: node.in_graph for name, node in self.nodes.items()}, 'edges':{name: {'score': float(edge.score), 'in_graph':edge.in_graph} for name, edge in self.edges.items()}, 'n_pos':self.n_pos}
        with open(filename, 'w') as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        if d['n_pos'] is None:
            g = Graph.from_model(d['cfg'])
        else:
            g = Graph.from_model_positional(d['cfg'], d['n_pos'])
        for name, in_graph in d['nodes'].items():
            g.nodes[name].in_graph = in_graph
        
        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g
    
    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False
        
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False 
            
        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.3,
        layout: str="dot",
        seed: Optional[int] = None
    ) -> pgv.AGraph:
        """
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name, 
                        fillcolor=colors[node.name], 
                        color="black", 
                        style="filled, rounded",
                        shape="box", 
                        fontname="Helvetica",
                        )

        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(max(minimum_penwidth, score) * 2),
                        color=edge.get_color(),
                        )
        return g
