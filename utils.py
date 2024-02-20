from typing import Tuple

import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F

def kl_div(logits, clean_logits, input_length, labels, mean=True):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    clean_logits = clean_logits[idx, input_length - 1]

    logprobs = torch.log_softmax(logits, dim=-1)
    clean_logprobs = torch.log_softmax(clean_logits, dim=-1)

    results = torch.nn.functional.kl_div(logprobs, clean_logprobs, log_target=True, reduction='none')
    return results.mean() if mean else results

def precision_at_k(clean_logits, corrupted_logits, input_length, labels, k=1, mean=True):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)
    predictions = torch.argmax(clean_probs, dim=-1).cpu()

    results = []
    for i, (ls,_) in enumerate(labels):
        r = torch.sum((ls == predictions[i]).float())
        results.append(r)
    results = torch.stack(results)
    return results.mean() if mean else results

def prob_diff_hypernymy(clean_logits, corrupted_logits, input_length, labels, mean=True, loss=False, logits=False):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if logits:
        clean_probs = clean_logits

    results = []
    for i, (ls,corrupted_ls) in enumerate(labels):
        r = clean_probs[i][ls.to(clean_probs.device)].sum() - clean_probs[i][corrupted_ls.to(clean_probs.device)].sum()
        results.append(r)
    results = torch.stack(results)
    if loss: 
        results = -results
    return results.mean() if mean else results

def batch(iterable, n:int=1):
   current_batch = []
   for item in iterable:
       current_batch.append(item)
       if len(current_batch) == n:
           yield current_batch
           current_batch = []
   if current_batch:
       yield current_batch


def inflow_outflow_difference(g, absolute:bool=True):
    diffs = []
    for name, node in g.nodes.items():
        if 'logits' in name or 'input' in name:
            continue
        diff = sum(edge.score for edge in node.child_edges) - sum(edge.score for edge in node.parent_edges)
        if absolute:
            diff = abs(diff)
        diffs.append(diff)
    diffs = np.array(diff)
    logit_inflow = sum(edge.score for edge in g.logits[0].parent_edges)
    input_outflow = sum(edge.score for edge in g.nodes['input'].child_edges)
    return diffs.mean(), logit_inflow, input_outflow