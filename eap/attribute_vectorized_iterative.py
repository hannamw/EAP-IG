from typing import Callable, List, Union, Optional, Literal

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node

def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int):
    input_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
    parent_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')
    
    input_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
    parent_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')

    child_gradients = torch.zeros((batch_size, n_pos, model.cfg.n_layers + 1, model.cfg.d_model), device='cuda')

    # normally would be batch_size, n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv'), model.cfg.d_model)
    # but the second n_pos is gone because those gradients don't exist anyway
    attn_child_gradients = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv'), model.cfg.d_model), device='cuda')

    processed_nodes = set()
    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def make_hook(t:torch.Tensor, index):
        def hook_fn(activations, hook):
            acts = activations.detach()
            try:
                t[index] += acts
            except RuntimeError as e:
                print(hook.name)
                print(t.size(), acts.size())
                raise e
        return hook_fn
    
    def node_to_matrix_slice(node: Node, matrix:Union[Literal['clean'], Literal['corrupted'], Literal['gradient']], qkv:Optional[Union[Literal['q'],Literal['k'],Literal['v']]]=None, group_heads:bool=True):
        if isinstance(node, InputNode):
            if matrix  == 'clean': 
                return input_activations_clean, slice(None)
            elif matrix == 'corrupted':
                return input_activations_corrupted, slice(None)
            else:
                raise ValueError(f'Matrix {matrix} invalid for InputNode')
        elif isinstance(node, LogitNode):
            if matrix == 'gradient':
                return child_gradients, (slice(None), slice(None), model.cfg.n_layers)
            else:
                raise ValueError(f'Matrix {matrix} invalid for LogitNode')
        elif isinstance(node, MLPNode):
            if matrix == 'clean':
                return parent_activations_clean, (slice(None), slice(None), node.layer, model.cfg.n_heads)
            elif matrix == 'corrupted':
                return parent_activations_corrupted, (slice(None), slice(None), node.layer, model.cfg.n_heads)
            elif matrix == 'gradient':
                return child_gradients, (slice(None), slice(None), node.layer)
            else:
                raise ValueError(f'Matrix {matrix} invalid for MLPNode')
        elif isinstance(node, AttentionNode):
            if matrix == 'clean':
                if group_heads:
                    return parent_activations_clean, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_heads))
                else:
                    return parent_activations_clean, (slice(None), slice(None), node.layer, node.head)
            elif matrix == 'corrupted':
                if group_heads:
                    return parent_activations_corrupted, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_heads))
                else:
                    return parent_activations_corrupted, (slice(None), slice(None), node.layer, node.head)
            elif matrix == 'gradient':
                assert qkv in 'qkv'
                return attn_child_gradients, (slice(None), slice(None), node.layer, slice(None), 'qkv'.index(qkv))
            else:
                raise ValueError(f'Matrix {matrix} invalid for AttentionNode')

    for name, node in graph.nodes.items():
        name_non_positional = name if graph.n_pos is None else  '_'.join(name.split('_')[:-1])
        if name_non_positional in processed_nodes:
            continue
        processed_nodes.add(name_non_positional)
        if isinstance(node, InputNode):
            fwd_hooks_clean.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'clean'))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'corrupted'))))
        elif isinstance(node, LogitNode):
            bwd_hooks.append((node.in_hook, make_hook(*node_to_matrix_slice(node, 'gradient'))))
        elif isinstance(node, MLPNode):
            fwd_hooks_clean.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'clean'))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'corrupted'))))
            bwd_hooks.append((node.in_hook, make_hook(*node_to_matrix_slice(node, 'gradient'))))
        elif isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue 
            processed_attn_layers.add(node.layer)
            fwd_hooks_clean.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'clean'))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(*node_to_matrix_slice(node, 'corrupted'))))
            for i, letter in enumerate('qkv'):
                bwd_hooks.append((node.qkv_inputs[i], make_hook(*node_to_matrix_slice(node, 'gradient', letter))))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        
    def make_input_construction_hook(node: Node, qkv:Optional[Union[Literal['q'],Literal['k'],Literal['v']]]=None):
        def input_construction_hook(activations, hook):
            activations = activations.clone()
            for edge in node.parent_edges:
                if edge.qkv != qkv:
                    continue

                parent:Node = edge.parent
                # reversed: we uncorrupt what's not in the circuit
                if not edge.in_graph:
                    clean_matrix, clean_slice = node_to_matrix_slice(parent, 'clean', group_heads=False)
                    corrupted_matrix, corrupted_slice = node_to_matrix_slice(parent, 'corrupted', group_heads=False)
                    try:
                        activations[edge.index] -= corrupted_matrix[corrupted_slice]
                        activations[edge.index] += clean_matrix[clean_slice]
                    except Exception as e:
                        print(parent.name, edge.name)
                        print(edge.index, corrupted_slice, clean_slice)
                        print(activations.size(), corrupted_matrix.size(), clean_matrix.size())
                        raise e
            return activations
        return input_construction_hook

    input_construction_hooks = []
    for node in graph.nodes.values():
        if isinstance(node, InputNode) or not node.in_graph:
            pass
        elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
            input_construction_hooks.append((node.in_hook, make_input_construction_hook(node)))
        elif isinstance(node, AttentionNode):
            for i, letter in enumerate('qkv'):
                input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter)))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (input_activations_corrupted, parent_activations_corrupted, input_activations_clean, parent_activations_clean, child_gradients, attn_child_gradients), input_construction_hooks

def get_activations(model: HookedTransformer, graph: Graph, clean_inputs: List[str], corrupted_inputs: List[str], metric: Callable[[Tensor], Tensor], labels, first_run=False):
    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_input_lengths(model, clean_inputs)

    (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (input_activations_corrupted, parent_activations_corrupted, input_activations_clean, parent_activations_clean, child_gradients, attn_child_gradients), input_construction_hooks = make_hooks_and_matrices(model, graph, batch_size, n_pos)

    if first_run:
        with model.hooks(fwd_hooks=fwd_hooks_corrupted), torch.inference_mode():
            corrupted_logits = model(corrupted_inputs)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_inputs)
            metric_value = metric(logits, corrupted_logits, input_lengths, labels)
            metric_value.backward()

        input_activation_differences = input_activations_corrupted - input_activations_clean
        parent_activation_differences = parent_activations_corrupted - parent_activations_clean
    else:
        with model.hooks(fwd_hooks=fwd_hooks_corrupted), torch.inference_mode():
            corrupted_logits = model(corrupted_inputs)

        with model.hooks(fwd_hooks=fwd_hooks_clean), torch.inference_mode():
            _ = model(clean_inputs)

        input_activation_differences = input_activations_corrupted - input_activations_clean
        parent_activation_differences = parent_activations_corrupted - parent_activations_clean

        with model.hooks(fwd_hooks=fwd_hooks_corrupted + input_construction_hooks, bwd_hooks=bwd_hooks):
            logits = model(corrupted_inputs)
            metric_value = metric(logits, corrupted_logits, input_lengths, labels)
            metric_value.backward()

    return input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients

def get_activations_ig(model: HookedTransformer, graph: Graph, clean_inputs: List[str], corrupted_inputs: List[str], metric: Callable[[Tensor], Tensor], labels, steps=50, first_run=False):
    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_input_lengths(model, clean_inputs)

    (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (input_activations_corrupted, parent_activations_corrupted, input_activations_clean, parent_activations_clean, child_gradients, attn_child_gradients), input_construction_hooks = make_hooks_and_matrices(model, graph, batch_size, n_pos)

    with torch.inference_mode():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_inputs)

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_inputs)

    input_activation_differences = input_activations_corrupted - input_activations_clean
    parent_activation_differences = parent_activations_corrupted - parent_activations_clean

    input_acts_clean, input_acts_corrupted = input_activations_clean.clone(), input_activations_corrupted.clone()

    def input_interpolation_hook(k: int):
        def hook_fn(activations, hook):
            return input_acts_clean + (k / steps) * (input_acts_corrupted - input_acts_clean) 
        return hook_fn

    total_steps = 0
    for step in range(1, steps+1):
        total_steps += 1
        if first_run:
            with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(corrupted_inputs)
                metric_value = metric(logits, clean_logits, input_lengths, labels)
                metric_value.backward()
        else:
            with torch.inference_mode(), model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre",input_interpolation_hook(step)), *fwd_hooks_clean]):
                _ = model(clean_inputs)

            with model.hooks(fwd_hooks=fwd_hooks_corrupted + input_construction_hooks, bwd_hooks=bwd_hooks):
                logits = model(corrupted_inputs)
                metric_value = metric(logits, clean_logits, input_lengths, labels)
                metric_value.backward()

    child_gradients = child_gradients / total_steps
    attn_child_gradients = attn_child_gradients / total_steps

    return input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients

allowed_aggregations = {'sum', 'mean', 'l2'}        
def attribute_vectorized_iterative_(model: HookedTransformer, graph: Graph, clean_inputs: Union[List[str], List[List[str]]], corrupted_inputs: Union[List[str], List[List[str]]], labels, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None, first_run=False):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    # component
    input_child_score_matrix = torch.zeros((model.cfg.n_layers + 1), device='cuda')
    # layer head qkv
    input_attn_score_matrix = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')
    # head component
    parent_child_score_matrix = torch.zeros((model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers + 1), device='cuda')
    # layer1 head1 layer2 head2 qkv
    parent_attn_score_matrix = torch.zeros((model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')

    if isinstance(clean_inputs[0], str):
        clean_inputs = [clean_inputs]
    if isinstance(corrupted_inputs[0], str):
        corrupted_inputs = [corrupted_inputs]

    total_items = sum(len(c) for c in clean_inputs)
    for clean, corrupted, label in tqdm(zip(clean_inputs, corrupted_inputs, labels), total=len(clean_inputs)):
        batch_size = len(clean)
        
        if integrated_gradients is None:
            input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients = get_activations(model, graph, clean, corrupted, metric, label, first_run=first_run)
        else:
            assert integrated_gradients > 0, f"the integrated_gradients argument gives (positive # steps (m) for the IG method, but got {integrated_gradients}"
            input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients = get_activations_ig(model, graph, clean, corrupted, metric, label, steps=integrated_gradients, first_run=first_run)

        if aggregation == 'sum' or aggregation == 'mean':
            denom = 1 if aggregation == 'sum' else model.cfg.d_model
            input_child_scores = einsum(input_activation_differences, child_gradients, 'batch pos hidden, batch pos component hidden -> component') / denom
            input_attn_scores = einsum(input_activation_differences, attn_child_gradients, 'batch pos hidden, batch pos layer head qkv hidden -> layer head qkv') / denom
            parent_child_scores = einsum(parent_activation_differences, child_gradients, 'batch pos layer head hidden, batch pos component hidden -> layer head component') / denom
            parent_attn_scores = einsum(parent_activation_differences, attn_child_gradients, 'batch pos layer1 head1 hidden, batch pos layer2 head2 qkv hidden -> layer1 head1 layer2 head2 qkv') / denom
        elif aggregation == 'l2':
            input_child_scores = torch.linalg.vector_norm(einsum(input_activation_differences, child_gradients, 'batch pos hidden, batch pos component hidden -> component hidden'), ord=2, dim=-1)
            input_attn_scores = torch.linalg.vector_norm(einsum(input_activation_differences, attn_child_gradients, 'batch pos hidden, batch pos layer head qkv hidden -> layer head qkv hidden'), ord=2, dim=-1)
            parent_child_scores = torch.linalg.vector_norm(einsum(parent_activation_differences, child_gradients, 'batch pos layer head hidden, batch pos component hidden -> layer head component hidden'), ord=2, dim=-1)
            parent_attn_scores = torch.linalg.vector_norm(einsum(parent_activation_differences, attn_child_gradients, 'batch pos layer1 head1 hidden, batch pos layer2 head2 qkv hidden -> layer1 head1 layer2 head2 qkv hidden'), ord=2, dim=-1)
        else:
            raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
        input_child_score_matrix += input_child_scores * batch_size / total_items
        input_attn_score_matrix += input_attn_scores * batch_size / total_items 
        parent_child_score_matrix += parent_child_scores * batch_size / total_items
        parent_attn_score_matrix += parent_attn_scores * batch_size / total_items

    input_child_score_matrix = input_child_score_matrix.cpu().numpy()
    input_attn_score_matrix = input_attn_score_matrix.cpu().numpy() 
    parent_child_score_matrix = parent_child_score_matrix.cpu().numpy()
    parent_attn_score_matrix = parent_attn_score_matrix.cpu().numpy() 

    qkv_map = {letter:i for i, letter in enumerate('qkv')}

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        if isinstance(edge.parent, InputNode):
            if isinstance(edge.child, AttentionNode):
                edge.score = input_attn_score_matrix[edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers

                edge.score = input_child_score_matrix[component]
        else:
            parent_head = edge.parent.head if isinstance(edge.parent, AttentionNode) else model.cfg.n_heads
            if isinstance(edge.child, AttentionNode):
                edge.score = parent_attn_score_matrix[edge.parent.layer, parent_head, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers

                edge.score = parent_child_score_matrix[edge.parent.layer, parent_head, component]

def attribute_vectorized_iterative(model: HookedTransformer, graph: Graph, clean_inputs: Union[List[str], List[List[str]]], corrupted_inputs: Union[List[str], List[List[str]]], labels, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None, iterations=[100,100,100,100]):
    for i, iteration in enumerate(iterations):
        attribute_vectorized_iterative_(model, graph, clean_inputs, corrupted_inputs, labels, metric, aggregation, integrated_gradients, first_run=(i==0))
        graph.apply_greedy(iteration, reset=(i==0))
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)