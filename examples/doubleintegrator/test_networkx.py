# importing networkx
import networkx as nx
# importing matplotlib.pyplot
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import itertools

G = nx.balanced_tree(4, 3)
pos = nx.nx_agraph.graphviz_layout(G, prog="circo", args="")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.axis("equal")
plt.show()

# # g = nx.Graph()

# # g.add_edge(1, 2)
# # g.add_edge(2, 3)
# # g.add_edge(3, 4)
# # g.add_edge(1, 4)
# # g.add_edge(1, 5)

# # g.add_edges_from([
# #     (1,2), (1,3),
# #     (2,4), (2,5), (3,6), (3,7)
# # ])
# subset_sizes = [1,2,4]
# subset_color = [
#     "gold",
#     "violet",
#     "violet",
#     "violet",
#     "violet",
#     "limegreen",
#     "limegreen",
#     "darkorange",
# ]


# def multilayered_graph(*subset_sizes):
#     extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
#     layers = [range(start, end) for start, end in extents]
#     G = nx.Graph()
#     for (i, layer) in enumerate(layers):
#         G.add_nodes_from(layer, layer=i)
#     for layer1, layer2 in nx.utils.pairwise(layers):
#         G.add_edges_from(itertools.product(layer1, layer2))
#     return G

# # nx.draw_planar(g, with_labels = True)
# # pos = graphviz_layout(g, prog="twopi")

# # G = multilayered_graph(*subset_sizes)

# G = nx.Graph()
# G.add_nodes_from([1], layer=1)
# G.add_nodes_from([2,3], layer=2)
# G.add_nodes_from([4,5,6,7], layer=3)
# G.add_edges_from([
#     (1,2), (1,3),
#     (2,4), (2,5), (3,6), (3,7)
# ])

# color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
# pos = nx.multipartite_layout(G, subset_key="layer")
# plt.figure(figsize=(8, 8))
# nx.draw(G, pos, node_color=color, with_labels=False)
# plt.axis("equal")
# plt.show()