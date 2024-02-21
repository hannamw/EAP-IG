from typing import Callable, List, Union, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int):
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
    gradients = torch.zeros((batch_size, n_pos, graph.n_backward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(t: torch.Tensor, index, add=True):
        def hook_fn(activations, hook):
            acts = activations.detach()
            try:
                if add:
                    t[index] += acts
                else:
                    t[index] -= acts
            except RuntimeError as e:
                print(hook.name, t.size(), acts.size())
                raise e
        return hook_fn

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        if not isinstance(node, LogitNode):
            fwd_index = (slice(None), slice(None), graph.forward_index(node))
            fwd_hooks_corrupted.append((node.out_hook, activation_hook(activation_difference, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, activation_hook(activation_difference, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = (slice(None), slice(None), graph.backward_index(node, qkv=letter))
                    bwd_hooks.append((node.qkv_inputs[i], activation_hook(gradients, bwd_index)))
            else:
                bwd_index = (slice(None), slice(None), graph.backward_index(node))
                bwd_hooks.append((node.in_hook, activation_hook(gradients, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (activation_difference, gradients)

def get_activations(model: HookedTransformer, graph: Graph, clean_inputs: torch.Tensor, corrupted_inputs: torch.Tensor, labels: torch.Tensor, positions: torch.Tensor, flags_tensor: torch.Tensor, metric: Callable[[Tensor], Tensor]):
    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_input_lengths(model, clean_inputs)

    (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (activation_difference, gradients) = make_hooks_and_matrices(model, graph, batch_size, n_pos)

    with model.hooks(fwd_hooks=fwd_hooks_corrupted):
        corrupted_logits = model(corrupted_inputs)

    with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
        logits = model(clean_inputs)
        metric_value = metric(logits, corrupted_logits, labels, positions, flags_tensor)
        metric_value.backward()

    return activation_difference, gradients

def get_activations_ig(model: HookedTransformer, graph: Graph, clean_inputs: torch.Tensor, corrupted_inputs: torch.Tensor, labels: torch.Tensor, positions: torch.Tensor, flags_tensor: torch.Tensor, metric: Callable[[Tensor], Tensor], steps=30):
    batch_size = clean_inputs.size(0)
    n_pos = clean_inputs.size(1)
    #n_pos, input_lengths = get_npos_input_lengths(model, clean_inputs)

    (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), (activation_difference, gradients) = make_hooks_and_matrices(model, graph, batch_size, n_pos)

    with torch.inference_mode():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_inputs)

        input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_inputs)

        input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

    def input_interpolation_hook(k: int):
        def hook_fn(activations, hook):
            new_input = input_activations_clean + (k / steps) * (input_activations_corrupted - input_activations_clean) 
            new_input.requires_grad = True 
            return new_input
        return hook_fn

    total_steps = 0
    for step in range(1, steps+1):
        total_steps += 1
        with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
            logits = model(clean_inputs)
            metric_value = metric(logits, clean_logits, labels, positions, flags_tensor)
            metric_value.backward()

    gradients = gradients / total_steps

    return activation_difference, gradients

allowed_aggregations = {'sum', 'mean', 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    all_scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    for clean, corrupted, label, positions, flags_tensor in tqdm(dataloader):
        total_items += len(clean)
        
        if integrated_gradients is None:
            activation_differences, gradients = get_activations(model, graph, clean, corrupted, label, positions, flags_tensor, metric)
        else:
            assert integrated_gradients > 0, f"integrated_gradients gives positive # steps (m), but got {integrated_gradients}"
            activation_differences, gradients = get_activations_ig(model, graph, clean, corrupted, label, positions, flags_tensor, metric, steps=integrated_gradients)

        scores = einsum(activation_differences, gradients,'batch pos n_forward hidden, batch pos n_backward hidden -> n_forward n_backward')

        if aggregation == 'mean':
            scores /= model.cfg.d_model
        elif aggregation == 'l2':
            scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)
        
        all_scores += scores #* batch_size / total_items
    all_scores /= total_items 
    all_scores = all_scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = all_scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]