from typing import Callable, List, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum

from .graph import Graph, LogitNode, AttentionNode

def tokenize_plus(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor]):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If you pass in None, we assume that you're using these hooks / matrices for evaluation only (so don't use the backwards hooks!)

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
    """
    separate_activations = model.cfg.use_normalization_before_and_after and scores is None
    if separate_activations:
        activation_difference = torch.zeros((2, batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)


    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
        
    # Fills up the activation difference matrix. In the default case (not separate_activations), 
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        try:
            if separate_activations:
                if add:
                    activation_difference[0, :, :, index] += acts
                else:
                    activation_difference[1, :, :, index] += acts
            else:
                if add:
                    activation_difference[:, :, index] += acts
                else:
                    activation_difference[:, :, index] -= acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(prev_index, bwd_index, scores.size(), s.size())
            raise e
    
    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    
    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
        
    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def compute_mean_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, per_position=False):
    """
    Compute the mean activations of a graph's nodes over a dataset.
    """
    def activation_hook(index, activations, hook, means=None, input_lengths=None):
        # defining a hook that will fill up our means tensor. Means is of shape
        # (n_pos, graph.n_forward, model.cfg.d_model) if per_position is True, otherwise
        # (graph.n_forward, model.cfg.d_model) 
        acts = activations.detach()

        # if you gave this hook input lengths, we assume you want to mean over positions
        if input_lengths is not None:
            mask = torch.zeros_like(activations)
            # mask out all padding positions
            mask[torch.arange(activations.size(0)), input_lengths - 1] = 1
            
            # we need ... because there might be a head index as well
            item_means = einsum(acts, mask, 'batch pos ... hidden, batch pos ... hidden -> batch ... hidden')
            
            # mean over the positions we did take, position-wise
            if len(item_means.size()) == 3:
                item_means /= input_lengths.unsqueeze(-1).unsqueeze(-1)
            else:
                item_means /= input_lengths.unsqueeze(-1)

            means[index] += item_means.sum(0)
        else:
            means[:, index] += acts.sum(0)

    # we're going to get all of the out hooks / indices we need for making hooks
    # but we can't make them until we have input length masks
    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)
        
        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    means_initialized = False
    total = 0
    for batch in tqdm(dataloader, desc='Computing mean'):
        # maybe the dataset is given as a tuple, maybe its just raw strings
        batch_inputs = batch[0] if isinstance(batch, tuple) else batch
        tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, batch_inputs, max_length=512)
        total += len(batch_inputs)

        if not means_initialized:
            # here is where we store the means
            if per_position:
                means = torch.zeros((n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            else:
                means = torch.zeros((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            means_initialized = True

        if per_position:
            input_lengths = None
        add_to_mean_hooks = [(hook_point, partial(activation_hook, index, means=means, input_lengths=input_lengths)) for hook_point, index in hook_points_indices]

        with model.hooks(fwd_hooks=add_to_mean_hooks):
            model(tokens, attention_mask=attention_mask)

    means = means.squeeze(0)
    means /= total
    return means if per_position else means.mean(0)


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(f"Number of positions must match, but do not: {n_pos} (clean) != {n_pos_corrupted} (corrupted)")
            print(clean)
            print(corrupted)
            raise ValueError("Number of positions must match")

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
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
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()
            
            if torch.isnan(scores).any().item():
                print("Metric value is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f'Step: {step}')
                raise ValueError("Metric value is NaN")

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', steps=30, intervention_dataloader: Optional[DataLoader]=None, quiet=False):

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

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

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    """Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        _type_: _description_
    """

    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

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

allowed_aggregations = {'sum', 'mean'}#, 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], method: Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations'], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', aggregation='sum', ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader, quiet=quiet)
    elif method == 'EAP-IG-inputs':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention, intervention_dataloader=intervention_dataloader, quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations'], but got {method}")


    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)

