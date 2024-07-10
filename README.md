# EAP-IG
This repo contains an implementation of [Edge Attribution Patching with Integrated Gradients (EAP-IG)](https://arxiv.org/abs/2403.17806), inspired by the [original paper](https://arxiv.org/abs/1703.01365) on integrated gradients for gradient-based input attribution. It allows you to efficiently score edges by their importance to a task, and define circuits based on those scores. EAP-IG is based on Edge Attribution Patching (EAP; see this [blog post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) and [this paper](https://arxiv.org/abs/2310.10348) for details). Compared to EAP, EAP-IG finds more faithful circuits, and for the use of KL divergence as a metric. Other fun additions in this repo include finding circuits in a scored computational graph using greedy search, and evaluating the circuit you've found, to test its faithfulness. For more details on EAP-IG, see [Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms
](https://arxiv.org/abs/2403.17806)

For a small demo of all these features, check out `greater_than.ipynb`. This repo is a work in progress, but feel free to contact me with any questions!

This EAP implementation contains the following files:
- `graph.py` contains the Node, Edge, and Graph classes.
- `attribute.py` contains the implementation of EAP/-IG
- `evaluate.py` contains code for evaluating circuits
- `visualization.py` contains code for choosing colors / controlling how circuits are visualized
- `utils.py` contains utils

Note that the repo is now intended to work with `transformer-lens=2.0.0` and will also probably work with most recent-ish `1.X.Y` versions. Because of a bug with `attention.hook_result`, I can't yet upgrade to the newest version of TransformerLens (but would like to do this soon)!

This repo owes a lot to:
- [The original ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), in particular for its conceptualization of the graph and its visualization—go check it out!
- [Aaquib Syed's original EAP implementation](https://github.com/Aaquib111/edge-attribution-patching/tree/minimal-implementation), for its memory efficient implementation of EAP

I recently updated this repo with a few improvements (v0.2.0); for the old version of this repo, please check out branch 0.1.0. The changes made are the following:
- Added a bunch of variants on EAP-IG, including the following. Find a comparison of these methods in [my paper introducing EAP-IG](https://arxiv.org/abs/2403.17806); long story short, either `EAP-IG` or `clean-corrupted` is probably best:
    - EAP, without any integrated gradients (`EAP`)
    - EAP-IG, where you interpolate between the clean and corrupted inputs (`EAP-IG`; this is my method)
    - EAP-IG, where you just average the gradients on the clean and corrupted example (`clean-corrupted`; this is a good baseline suggested by Neel Nanda)
    - EAP-IG, where you set the output to each node to interpolate between its original output and entirely corrupted activations (`EAP-IG-partial-activations`). This is (by my understanding) equivalent to the [`mask_gradient_prune_scores` method from AutoCircuit](https://ufo-101.github.io/auto-circuit/reference/prune_algos/mask_gradient/#auto_circuit.prune_algos.mask_gradient.mask_gradient_prune_scores).
    - EAP-IG, where you set the output of each node to interpolate between entirely clean and entirely corrupted activations(`EAP-IG-activations`); this is the EAP-IG introduced in [Marks et al.'s (2024) paper on feature circuits](https://arxiv.org/abs/2403.19647). Note that this version is slower, as it iterates over sublayers (attention blocks and MLPs). Ablating all the sublayers at once doesn't work, so I haven't made it available as an option (it's in the code though).
- Changed how you specify which variant you want to use. Now, when you call `attribute`, just set the argument `method` to one of this above; to specify the number of steps, set `ig_steps` (default is 5). The default `method` is `EAP-IG`
- Eliminated the fast but memory-hungry versions of EAP-IG originally contained in `attribute.py`; I didn't want to update two versions, and memory efficiency matters more. What was once `attribute_mem.py` is now `attribute.py`.
- `eap.evaluate_graph` is now just `eap.evaluate`
- Graph evaluation is now tensorized (and skips unnecessary nodes), making it much faster.
- Changed how tokenizers are handled to be more compatible with newer versions of TransformerLens.
- You can now export graphs as `.pt` files, which takes less space than `.json` files.
- Added a `requirements.txt` file, as well as an `environment.yml` file. I used conda to handle my virtual environments, so the latter might work better.
- Removed the dependency on `cmapy`, because cmapy isn't updated to work with newer versions of matplotlib; full disclosure, some of the `visualization.py` code thus comes directly from `cmapy`, just updated a little. This is also nice because it removes the need to download `cmapy`'s dependencies.
- Added more comments (but more documentation is still needed!)