from typing import Callable, List, Union

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

def get_activations(model: HookedTransformer, graph: Graph, clean_inputs, corrupted_inputs, labels,  metric: Callable[[Tensor], Tensor]):
    fwd_names = graph.parent_node_names()
    fwd_filter = lambda x: x in fwd_names
    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    with model.hooks(fwd_hooks=corrupted_fwd_hooks):
        _ = model(corrupted_inputs)

    clean_fwd_cache, clean_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)

    bwd_names = graph.child_node_names()
    bwd_filter = lambda x: x in bwd_names
    clean_bwd_cache, _, clean_bwd_hooks = model.get_caching_hooks(bwd_filter, incl_bwd=True)

    with model.hooks(fwd_hooks=clean_fwd_hooks, bwd_hooks=clean_bwd_hooks):
        logits = model(clean_inputs)
        metric_value = metric(logits, labels)
        metric_value.backward()

    return corrupted_fwd_cache, clean_fwd_cache, clean_bwd_cache

# forward is for parent, backward is for child
def attribute(model: HookedTransformer, graph: Graph, clean_inputs:List[str], corrupted_inputs:List[str], labels, metric: Callable[[Tensor], Tensor]):
    corrupted_fwd_cache, clean_fwd_cache, clean_bwd_cache = get_activations(model, graph, clean_inputs, corrupted_inputs, labels, metric)
    for node in tqdm(graph.nodes.values(), total=len(graph.nodes)):
        if not node.children:
            continue
        parent = node
        edges = parent.child_edges
        
        parent_activation_clean = clean_fwd_cache[parent.out_hook][parent.index]
        parent_activation_corrupted = corrupted_fwd_cache[parent.out_hook][parent.index]

        if torch.all(parent_activation_clean == parent_activation_corrupted):
            for edge in edges:
                edge.score = 0
            continue

        parent_activation_difference = parent_activation_corrupted - parent_activation_clean
        for edge in edges:
            try:
                child_gradient = clean_bwd_cache[edge.hook + "_grad"][edge.index]

                score = (parent_activation_difference * child_gradient).sum().cpu().item()
                edge.score = score
            except RuntimeError as e:
                print(f'Failed on {edge}')
                raise e
            
def attribute_vectorized(model: HookedTransformer, graph: Graph, clean_inputs: Union[List[str], List[List[str]]], corrupted_inputs: Union[List[str], List[List[str]]], labels, metric: Callable[[Tensor], Tensor]):

    if isinstance(clean_inputs[0], str):
        clean_inputs = [clean_inputs]
    if isinstance(corrupted_inputs[0], str):
        corrupted_inputs = [corrupted_inputs]

    n_pos = 1 + len(model.tokenizer(clean_inputs[0][0])[0])

    # pos component
    input_child_score_matrix = torch.zeros((n_pos, model.cfg.n_layers + 1), device='cuda')
    # pos end layer head qkv
    input_attn_score_matrix = torch.zeros((n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')
    # pos layer head component
    parent_child_score_matrix = torch.zeros((n_pos, model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers + 1), device='cuda')
    # pos end layer1 head1 layer2 head2 qkv
    parent_attn_score_matrix = torch.zeros((n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')

    total_items = sum(len(c) for c in clean_inputs)
    for clean, corrupted, label in tqdm(zip(clean_inputs, corrupted_inputs, labels), total=len(clean_inputs)):
        batch_size = len(clean)
        corrupted_fwd_cache, clean_fwd_cache, clean_bwd_cache = get_activations(model, graph, clean, corrupted, label, metric)

        input_activation_differences = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')

        parent_activation_differences = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')

        child_gradients = torch.zeros((batch_size, n_pos, model.cfg.n_layers + 1, model.cfg.d_model), device='cuda')

        attn_child_gradients = torch.zeros((batch_size, n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv'), model.cfg.d_model), device='cuda')

        processed_heads = set()
        for node in graph.nodes.values():
            if isinstance(node, InputNode):
                parent_activation_clean = clean_fwd_cache[node.out_hook][node.index]
                parent_activation_corrupted = corrupted_fwd_cache[node.out_hook][node.index]

                input_activation_differences[:, node.pos] = parent_activation_corrupted - parent_activation_clean
                
            elif isinstance(node, LogitNode):
                child_gradient = clean_bwd_cache[node.in_hook + "_grad"][node.index]

                child_gradients[:, node.pos, model.cfg.n_layers] = child_gradient
                
            else:
                parent_activation_clean = clean_fwd_cache[node.out_hook][node.index]
                parent_activation_corrupted = corrupted_fwd_cache[node.out_hook][node.index]
                head = node.head if isinstance(node, AttentionNode) else model.cfg.n_heads

                parent_activation_differences[:, node.pos, node.layer, head] = parent_activation_corrupted - parent_activation_clean

                if isinstance(node, AttentionNode):
                    if (node.layer, node.head) in processed_heads:
                        continue
                    processed_heads.add((node.layer, node.head))
                    for i, letter in enumerate('qkv'):
                        hook = f'blocks.{node.layer}.hook_{letter}_input'
                        # batch, pos, hidden
                        child_gradient = clean_bwd_cache[hook + "_grad"][:, :, node.head]
                        #batch pos end layer2 head2 qkv hidden
                        child_gradient = child_gradient.unsqueeze(2)
                        attn_child_gradients[:, :, :, node.layer, node.head, i] = child_gradient

                elif isinstance(node, MLPNode):
                    child_gradient = clean_bwd_cache[node.in_hook + "_grad"][node.index]

                    child_gradients[:, node.pos, node.layer] = child_gradient
                else:
                    raise RuntimeError(f'Encountered invalid node: {node} of type {type(node)}')

        input_child_score_matrix += einsum(input_activation_differences, child_gradients, 'batch pos hidden, batch pos component hidden -> pos component') * batch_size / total_items

        input_attn_score_matrix += einsum(input_activation_differences, attn_child_gradients, 'batch pos hidden, batch pos end layer head qkv hidden -> pos end layer head qkv') * batch_size / total_items

        parent_child_score_matrix += einsum(parent_activation_differences, child_gradients, 'batch pos layer head hidden, batch pos component hidden -> pos layer head component') * batch_size / total_items

        parent_attn_score_matrix += einsum(parent_activation_differences, attn_child_gradients, 'batch pos layer1 head1 hidden, batch pos end layer2 head2 qkv hidden -> pos end layer1 head1 layer2 head2 qkv') * batch_size / total_items

    # input_child_score_matrix /= total_items #* len(clean_inputs)
    # input_attn_score_matrix /= total_items #* len(clean_inputs)
    # parent_child_score_matrix /= total_items #* len(clean_inputs)
    # parent_attn_score_matrix /= total_items #* len(clean_inputs)
    
    if not graph.n_pos:
        input_child_score_matrix = input_child_score_matrix.sum(0)
        input_attn_score_matrix = input_attn_score_matrix.sum(0).sum(0) / (model.cfg.n_heads + 1)
        parent_child_score_matrix = parent_child_score_matrix.sum(0) 
        parent_attn_score_matrix = parent_attn_score_matrix.sum(0).sum(0) / (model.cfg.n_heads + 1)

    input_child_score_matrix = input_child_score_matrix.cpu().numpy()
    input_attn_score_matrix = input_attn_score_matrix.cpu().numpy()
    parent_child_score_matrix = parent_child_score_matrix.cpu().numpy()
    parent_attn_score_matrix = parent_attn_score_matrix.cpu().numpy()

    qkv_map = {letter:i for i, letter in enumerate('qkv')}

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        if isinstance(edge.parent, InputNode):
            if isinstance(edge.child, AttentionNode):
                if graph.n_pos:
                    edge.score = input_attn_score_matrix[edge.parent.pos, edge.child.pos, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
                else:
                    edge.score = input_attn_score_matrix[edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers
                if graph.n_pos:
                    edge.score = input_child_score_matrix[edge.parent.pos, component]
                else:
                    edge.score = input_child_score_matrix[component]
        else:
            parent_head = edge.parent.head if isinstance(edge.parent, AttentionNode) else model.cfg.n_heads
            if isinstance(edge.child, AttentionNode):
                if graph.n_pos:
                    edge.score = parent_attn_score_matrix[edge.parent.pos, edge.child.pos, edge.parent.layer, parent_head, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
                else:
                    edge.score = parent_attn_score_matrix[edge.parent.layer, parent_head, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers
                if graph.n_pos:
                    edge.score = parent_child_score_matrix[edge.parent.pos, edge.parent.layer, parent_head, component]
                else:
                    edge.score = parent_child_score_matrix[edge.parent.layer, parent_head, component]