from typing import Callable, List, Union

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def get_activations(model: HookedTransformer, graph: Graph, clean_inputs: List[str], corrupted_inputs: List[str], metric: Callable[[Tensor], Tensor], labels):
    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_input_lengths(model, clean_inputs)

    input_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
    parent_activations_clean = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')
    
    input_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.d_model), device='cuda')
    parent_activations_corrupted = torch.zeros((batch_size, n_pos, model.cfg.n_layers, model.cfg.n_heads + 1 , model.cfg.d_model), device='cuda')

    child_gradients = torch.zeros((batch_size, n_pos, model.cfg.n_layers + 1, model.cfg.d_model), device='cuda')
    attn_child_gradients = torch.zeros((batch_size, n_pos, n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv'), model.cfg.d_model), device='cuda')

    processed_nodes = set()
    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def make_hook(t:torch.Tensor, index, unsqueeze=False):
        def hook_fn(activations, hook):
            acts = activations.detach()
            if unsqueeze:
                acts = acts.unsqueeze(2)
            try:
                t[index] = acts
            except RuntimeError as e:
                print(hook.name)
                print(t.size(), acts.size())
                raise e
        return hook_fn

    for name, node in graph.nodes.items():
        name_non_positional = name if graph.n_pos is None else  '_'.join(name.split('_')[:-1])
        if name_non_positional in processed_nodes:
            continue
        processed_nodes.add(name_non_positional)
        if isinstance(node, InputNode):
            fwd_hooks_clean.append((node.out_hook, make_hook(input_activations_clean, slice(None))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(input_activations_corrupted, slice(None))))
        elif isinstance(node, LogitNode):
            bwd_hooks.append((node.in_hook, make_hook(child_gradients, (slice(None), slice(None), model.cfg.n_layers))))
        elif isinstance(node, MLPNode):
            fwd_hooks_clean.append((node.out_hook, make_hook(parent_activations_clean, (slice(None), slice(None), node.layer, model.cfg.n_heads))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(parent_activations_corrupted, (slice(None), slice(None), node.layer, model.cfg.n_heads))))
            bwd_hooks.append((node.in_hook, make_hook(child_gradients, (slice(None), slice(None), node.layer))))
        elif isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue 
            processed_attn_layers.add(node.layer)
            fwd_hooks_clean.append((node.out_hook, make_hook(parent_activations_clean, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_heads)))))
            fwd_hooks_corrupted.append((node.out_hook, make_hook(parent_activations_corrupted, (slice(None), slice(None), node.layer, slice(0, model.cfg.n_heads)))))
            for i, letter in enumerate('qkv'):
                bwd_hooks.append((node.qkv_inputs[i], make_hook(attn_child_gradients, (slice(None), slice(None), slice(None), node.layer, slice(None), i), unsqueeze=True)))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    with model.hooks(fwd_hooks=fwd_hooks_corrupted):
        bad_logits = model(corrupted_inputs)

    with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
        logits = model(clean_inputs)
        #print(clean_inputs, corrupted_inputs)
        #print(logits.size(), bad_logits.size())
        #print(input_lengths)
        metric_value = metric(logits, bad_logits, input_lengths, labels)
        metric_value.backward()

        input_activation_differences = input_activations_corrupted - input_activations_clean
        parent_activation_differences = parent_activations_corrupted - parent_activations_clean

    return input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients
            
def attribute_vectorized(model: HookedTransformer, graph: Graph, clean_inputs: Union[List[str], List[List[str]]], corrupted_inputs: Union[List[str], List[List[str]]], labels, metric: Callable[[Tensor], Tensor]):
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
        input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients = get_activations(model, graph, clean, corrupted, metric, label)

        input_child_score_matrix += einsum(input_activation_differences, child_gradients, 'batch pos hidden, batch pos component hidden -> component') * batch_size / total_items

        input_attn_score_matrix += einsum(input_activation_differences, attn_child_gradients, 'batch pos hidden, batch pos end layer head qkv hidden -> end layer head qkv').mean(0) * batch_size / total_items 

        parent_child_score_matrix += einsum(parent_activation_differences, child_gradients, 'batch pos layer head hidden, batch pos component hidden -> layer head component') * batch_size / total_items

        parent_attn_score_matrix += einsum(parent_activation_differences, attn_child_gradients, 'batch pos layer1 head1 hidden, batch pos end layer2 head2 qkv hidden -> end layer1 head1 layer2 head2 qkv').mean(0) * batch_size / total_items

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

def attribute_vectorized_positional(model: HookedTransformer, graph: Graph, clean_inputs: Union[List[str], List[List[str]]], corrupted_inputs: Union[List[str], List[List[str]]], labels, metric: Callable[[Tensor], Tensor]):
    # pos component
    input_child_score_matrix = torch.zeros((graph.n_pos, model.cfg.n_layers + 1), device='cuda')
    # pos end layer head qkv
    input_attn_score_matrix = torch.zeros((graph.n_pos, graph.n_pos, model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')
    # pos layer head component
    parent_child_score_matrix = torch.zeros((graph.n_pos, model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers + 1), device='cuda')
    # pos end layer1 head1 layer2 head2 qkv
    parent_attn_score_matrix = torch.zeros((graph.n_pos, graph.n_pos, model.cfg.n_layers, model.cfg.n_heads + 1, model.cfg.n_layers, model.cfg.n_heads, len('qkv')), device='cuda')

    if isinstance(clean_inputs[0], str):
        clean_inputs = [clean_inputs]
    if isinstance(corrupted_inputs[0], str):
        corrupted_inputs = [corrupted_inputs]

    total_items = sum(len(c) for c in clean_inputs)
    for clean, corrupted, label in tqdm(zip(clean_inputs, corrupted_inputs, labels), total=len(clean_inputs)):
        batch_size = len(clean)
        input_activation_differences, parent_activation_differences, child_gradients, attn_child_gradients = get_activations(model, graph, clean, corrupted, metric, label)

        input_child_score_matrix += einsum(input_activation_differences, child_gradients, 'batch pos hidden, batch pos component hidden -> pos component') * batch_size / total_items

        input_attn_score_matrix += einsum(input_activation_differences, attn_child_gradients, 'batch pos hidden, batch pos end layer head qkv hidden -> pos end layer head qkv') * batch_size / total_items

        parent_child_score_matrix += einsum(parent_activation_differences, child_gradients, 'batch pos layer head hidden, batch pos component hidden -> pos layer head component') * batch_size / total_items

        parent_attn_score_matrix += einsum(parent_activation_differences, attn_child_gradients, 'batch pos layer1 head1 hidden, batch pos end layer2 head2 qkv hidden -> pos end layer1 head1 layer2 head2 qkv') * batch_size / total_items

    input_child_score_matrix = input_child_score_matrix.cpu().numpy()
    input_attn_score_matrix = input_attn_score_matrix.cpu().numpy()
    parent_child_score_matrix = parent_child_score_matrix.cpu().numpy()
    parent_attn_score_matrix = parent_attn_score_matrix.cpu().numpy()

    qkv_map = {letter:i for i, letter in enumerate('qkv')}

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        if isinstance(edge.parent, InputNode):
            if isinstance(edge.child, AttentionNode):
                edge.score = input_attn_score_matrix[edge.parent.pos, edge.child.pos, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers
                edge.score = input_child_score_matrix[edge.parent.pos, component]
        else:
            parent_head = edge.parent.head if isinstance(edge.parent, AttentionNode) else model.cfg.n_heads
            if isinstance(edge.child, AttentionNode):
                edge.score = parent_attn_score_matrix[edge.parent.pos, edge.child.pos, edge.parent.layer, parent_head, edge.child.layer, edge.child.head, qkv_map[edge.qkv]]
            else:
                component = edge.child.layer if isinstance(edge.child, MLPNode) else model.cfg.n_layers
                edge.score = parent_child_score_matrix[edge.parent.pos, edge.parent.layer, parent_head, component]
