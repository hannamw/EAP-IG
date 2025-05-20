# EAP-IG
This library contains various resources for finding circuits in autoregressive transformer LMs. At a high level, a circuit is the part of your model responsible for performing a given task; all nodes / edges outside the circuit can be corrupted without harming model performance. For more on circuits, see [this paper](https://arxiv.org/abs/2403.17806 ) or [this paper](https://arxiv.org/abs/2403.19647). For a demo of this library's features, check out `greater_than.ipynb`; for a demo using larger models (Llama-3 8B), check out `ioi.ipynb`.

This library has tools that will let you do a variety of things:
- Construct a `Graph` object representing the computational graph of most autoregressive transformer LMs in the [TransformerLens library](https://github.com/TransformerLensOrg/TransformerLens). Computational graphs can be drawn at the following levels:
    - **Node and edge (default)**: Nodes are model components (attention heads and MLPs), and edges are connections between them (across layers, via the residual stream)
    - **Node**: The graph contains only nodes, and we disregard the edges. This is equivalent to saying that for every node in the circuit, all of its outgoing edges are also in the circuit.
    - **Neuron**: The graph contains only nodes, split into neurons. That is, you can include individual neurons, or output dimensions of a given component. 
- Use attribution-based circuit-finding methods to produce scores (indirect effect estimates) for each node or edge in the computational graph. The attribution methods supported are:
    - [Edge Attribution Patching (EAP)](https://arxiv.org/abs/1703.01365): Computes a first-order approximation of the indirect effect of each edge, i.e. the amount that your loss changes upon corrupting the edge. Essentially multiplies the change in component outputs by the gradient on clean inputs. Runs in O(1) time. See the [original blog post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) for more info.
    - [Edge Attribution Patching with Integrated Gradients (EAP-IG, inputs)](https://arxiv.org/abs/2403.17806): An adaptation of EAP that improves circuit quality by averaging the gradient computation over *m* steps taken between the clean and corrupted inputs, as in the [integrated gradients paper](https://arxiv.org/abs/1703.01365). Takes O(*m*) time.
    - [Edge Attribution Patching with Integrated Gradients (EAP-IG, activations)](): Another adaptation of EAP using integrated gradients; instead of taking the gradient when the input embeddings are interpolated between the clean and corrupted inputs, it interpolates between the clean/corrupted activations for each component. This takes longer (O(*m * L*) time, given an *L*-layer model), but is somewhat more principled, and allows for estimating zero / mean ablation effects as well (just like EAP).
    - [Clean-Corrupted](https://arxiv.org/abs/2403.17806): A variant of EAP/-IG that takes the gradient at two steps: the clean and corrupted input.
- Use either a greedy-search or top-n approach to find a circuit of a given size based on these scores
- Evaluate your circuit's performance (allowing you to compute its faithfulness)

## How to install this library
To use this library, just install it using `pip install .`. If you'd like to be able to visualize the graphs you create, please use the `viz` option: `pip install .[viz]`. This may require you to install graphviz:

<details>
<summary>How to install graphviz</summary>

**MacOS**
```bash
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
pip install . # or `uv sync`
```

**Ubuntu**
```bash
apt-get update
apt-get install -y graphviz libgraphviz-dev build-essential
```

For other operating systems or if you encounter build errors, ensure the Graphviz C libraries are installed and accessible to the build system via environment variables (like `CFLAGS` and `LDFLAGS`).
</details>

## How to use this library
For a demo of this library's features, check out `greater_than.ipynb`; for a demo using larger models (Llama-3 8B), check out `ioi.ipynb`. In general, the circuit-finding pipeline looks like this:
- Define a task with clean and corrupted inputs, a label associated with the clean inputs, and a metric measuring model performance. (`dataloader = EAPDataset('greater-than').to_dataloader()`, `metric = ...`)
- Define your model's computation graph at the desired level of granularity. (`graph = Graph.from_model(model)`)
- Use an attribution method to estimate the change in the metric that would occur if you were to corrupted / mean-ablate / zero-ablate each unit in your computation graph (i.e., estimate each unit's indirect effect). (`attribute(model, graph, dataloader, metric, method='EAP-IG-inputs', ig_steps=5)`)
- Using the indirect effects / scores calculated, define a circuit by taking the top-n edges / nodes / neurons of your graph. (`graph.apply_topn(n)`)
- Evaluate your circuit's performance, recording the metric when you actually corrupt / ablate all edges / nodes / neurons not in the circuit. (`results = evaluate_graph(model, graph, dataloader, metric)`)

## FAQs
- **How is the computation graph drawn?**: In this library, graphs are defined as being collections of nodes and edges, where nodes are either the inputs, attention heads, MLPs, or logits. Edges connect nodes across layers, accounting for the fact that nodes can engage in cross-layer communication via the residual stream. Each MLP (and the logits) has 1 input, but each attention head has 3: the Q, K, and V input.
- **Which models are compatible with this library?**: In general, this library works with autoregressive transformer LMs in TransformerLens. It's important that models use pre-LayerNorm, as post-LayerNorm means that the residual stream is no longer a sum of all previous components. The models I have used so far are: GPT-2, Pythia, Mistral, Qwen, OLMo, Llama, and Gemma (using a workaround / hack since there is a post layer-norm that doesn't totally destroy the residual stream.)
- **What about models with Grouped Query Attention (GQA)?**: To work with these models, please ungroup the GQA by setting `model.cfg.ungroup_grouped_query_attention = True`; this will remove all of the efficiency benefits of GQA, but allow the model to be used with this library.
- **What about zero and mean ablations?**: I think these are often best avoided, at least zero-ablation. But these are supported as well (with EAP / EAP-IG (activations)). Just set the `intervention` argument of `attribute` and `evaluate_graph` to `zero`, `mean`, or `mean-positional`; in the latter case, all inputs must have the same length / structure. You can specify the dataloader to take the mean over via the `intervention_dataloader` argument.

## More Info
This library contains the following files:
- `graph.py` contains the Node, Edge, and Graph classes.
- `attribute.py` contains the implementation of EAP/-IG
- `attribute_node.py` contains the implementation of EAP/-IG, but for nodes / neurons
- `evaluate.py` contains code for evaluating circuits
- `visualization.py` contains code for choosing colors / controlling how circuits are visualized

This repo owes a lot to:
- [The original ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery), in particular for its conceptualization of the graph and its visualization—go check it out!
- [Aaquib Syed's original EAP implementation](https://github.com/Aaquib111/edge-attribution-patching/tree/minimal-implementation), for its memory efficient implementation of EAP