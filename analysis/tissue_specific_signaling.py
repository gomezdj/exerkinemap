import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define nodes for organs and immune environments
organs = ['Muscle', 'Heart', 'Intestine']
immune_environments = ['Spleen', 'Thymus', 'Kidney']

# Add nodes to the graph
G.add_nodes_from(organs + immune_environments)

# Define edges representing regulatory interactions (example interactions)
interactions = [
    ('Muscle', 'Heart'),
    ('Heart', 'Intestine'),
    ('Intestine', 'Spleen'),
    ('Spleen', 'Thymus'),
    ('Thymus', 'Kidney'),
    ('Kidney', 'Muscle'),
    ('Muscle', 'Spleen'),
    ('Heart', 'Kidney'),
]

# Add edges to the graph
G.add_edges_from(interactions)

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_weight='bold')
plt.title('Exerkine Regulatory Network')
plt.show()
