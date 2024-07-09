from typing import Callable, List, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum

from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: torch.Tensor, detach=True):
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        grads = gradients.detach()
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :fwd_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)#.to(scores.device)
            scores[:fwd_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), grads.size())
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index =  graph.forward_index(node)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, fwd_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_partial_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        def output_interpolation_hook(k: int, difference: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                new_output = activations + (1 - k / steps) * difference
                return new_output
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(node.out_hook, output_interpolation_hook(step, activation_difference[:, :, graph.forward_index(node)])) for node in graph.nodes.values() if not isinstance(node, LogitNode)], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_activations_all(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)


        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)


        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            fwd_hooks = []
            for node in graph.nodes.values():
                if not isinstance(node, LogitNode):
                    clean_acts = activations_clean[:, :, graph.forward_index(node)]
                    corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                    fwd_hooks.append((node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts)))

            with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)

                metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)


        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=False)


        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [[graph.nodes['input']]]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append([graph.nodes[f'a{layer}.h{head}'] for head in range(graph.cfg['n_heads'])])
            nodeslist.append([graph.nodes[f'm{layer}']])

        for nodes in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                fwd_hooks = []
                for node in nodes:
                    clean_acts = activations_clean[:, :, graph.forward_index(node)]
                    corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                    fwd_hooks.append((node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts)))

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens = model.to_tokens(clean, prepend_bos=True, padding_side='right')
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=True, padding_side='right')
        attention_mask = get_attention_mask(model.tokenizer, clean_tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)


        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean', 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], aggregation='sum', method: Union[Literal['EAP', 'EAP-IG', 'EAP-IG-partial-activations', 'EAP-IG-activations', 'clean-corrupted']]='EAP-IG', ig_steps: Optional[int]=5, quiet=False):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    if method != 'EAP' and method != 'clean-corrupted':
        assert ig_steps is not None, f"ig_steps must be set for method {method}"

    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG':
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'EAP-IG-partial-activations':
        scores = get_scores_ig_partial_activations(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method  == 'clean-corrupted':
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG', 'EAP-IG-partial-activations', 'EAP-IG-activations', 'clean-corrupted'], but got {method}")

    if aggregation == 'mean':
        scores /= model.cfg.d_model
    elif aggregation == 'l2':
        scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)
        
    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]