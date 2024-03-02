# EAP-positional
This repo contains an implementation of Edge Attribution Patching (EAP; see this blog post and this paper for details). This allows you to efficiently scores edges by their importance to a task, and define circuits based on those scores. This repo upgrades EAP into Edge Attribution Patching with Integrated Gradients (EAP-IG), inspired by the original integrated gradients paper for gradient-based input attribution. This allows for better scores, and for the use of KL divergence as a metric. Other fun additions include finding circuits in a scored computational graph using greedy search, and evaluating the circuit you've found, to test its faithfulness.

For a small demo of all these features, check out `greater_than.py`. This repo is a work in progress, and not really meant yet for external use, but feel free to contact me with any questions!

This repo contains the following files:
- `graph.py` contains the Node, Edge, and Graph classes.
- `attribute_mem.py` contains the code for the memory-efficient implementation (see https://github.com/Aaquib111/edge-attribution-patching/tree/minimal-implementation)
- `attribute.py` contains the code for the slightly faster, but memory-hungry implementation
- `evaluate.py` contains the code for evaluating circuits
- `greater_than.py` shows how to run EAP on the greater-than task. 
- `greater_than_dataset.py` contains an implementation of the greater-than task (copied pretty much verbatim from the original)

This repo owes a lot to [the original ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), in particular the conceptualization of the graph and its visualization—go check it out!
