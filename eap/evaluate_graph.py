from typing import Callable, List, Union
from functools import partial 

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum 

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node, Edge
from .attribute_mem import make_hooks_and_matrices

def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metrics: List[Callable[[Tensor], Tensor]], prune:bool=True, quiet=False):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes['logits'].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    in_graph_matrix = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)
    for edge in graph.edges.values():
        if edge.in_graph:
            in_graph_matrix[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = 1
            
    in_graph_matrix = 1 - in_graph_matrix

    # For each node in the graph, construct its input (in the case of attention heads, multiple inputs) by corrupting the incoming edges that are not in the circuit.
    # We assume that the corrupted cache is filled with corresponding corrupted activations, and that the mixed cache contains the computed activations from preceding nodes in this forward pass.
    def make_input_construction_hook(activation_differences, in_graph_vector, attn=False):
        def input_construction_hook(activations, hook):
            if attn:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous head -> batch pos head hidden')
            else:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous -> batch pos hidden')
            activations += update
            return activations
        return input_construction_hook

    # we make input construction hooks for every node but InputNodes. 
    # We can also skip nodes not in the graph; it doesn't matter what their outputs are, as they will always be corrupted / reconstructed when serving as inputs to other nodes.
    # AttentionNodes have 3 inputs to reconstruct
    def make_input_construction_hooks(activation_differences, in_graph_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # add attention hooks:
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], attn=True)))
            # add MLP hook
            if graph.nodes[f'm{layer}'].in_graph:
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_construction_hooks.append((node.in_hook, make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index])))
        # Logit node input construction
        node = graph.nodes[f'logits']
        if node.in_graph:
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_construction_hooks.append((node.in_hook, make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index])))
        return input_construction_hooks
            
    # and here we actually run / evaluate the model
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)
        
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, len(clean), n_pos, None)
        
        input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix)
        with torch.inference_mode():
            with model.hooks(fwd_hooks_corrupted):
                corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)

            if empty_circuit:
                logits = corrupted_logits
            else:
                with model.hooks(fwd_hooks_clean + input_construction_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)

        for i, metric in enumerate(metrics):
            r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_baseline(model: HookedTransformer, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False, quiet=False):
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    
    results = [[] for _ in metrics]
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in tqdm(dataloader):
        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)

        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            logits = model(clean_tokens, attention_mask=attention_mask)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results