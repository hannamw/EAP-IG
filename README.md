# EAP-IG
This repo contains an implementation of [Edge Attribution Patching with Integrated Gradients (EAP-IG)](https://arxiv.org/abs/2403.17806), inspired by the original integrated gradients paper for gradient-based input attribution. It allows you to efficiently scores edges by their importance to a task, and define circuits based on those scores. EAP-IG is based on Edge Attribution Patching (EAP; see this [blog post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) and [this paper](https://arxiv.org/abs/2310.10348) for details). Compared to EAP, EAP-IG finds more faithful circuits, and for the use of KL divergence as a metric. Other fun additions in this repo include finding circuits in a scored computational graph using greedy search, and evaluating the circuit you've found, to test its faithfulness. For more details on EAP-IG, see [Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms
](https://arxiv.org/abs/2403.17806)

For a small demo of all these features, check out `greater_than.ipynb`. This repo is a work in progress, but feel free to contact me with any questions!

This EAP implementation contains the following files:
- `graph.py` contains the Node, Edge, and Graph classes.
- `attribute_mem.py` contains a memory-efficient EAP/-IG implementation (see https://github.com/Aaquib111/edge-attribution-patching/tree/minimal-implementation)
- `attribute.py` contains a slightly faster, but memory-hungry EAP-IG implementation
- `evaluate.py` contains code for evaluating circuits

This repo owes a lot to [the original ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), in particular the conceptualization of the graph and its visualization—go check it out!
