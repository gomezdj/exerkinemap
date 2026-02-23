import numpy as np

def extract_features(diffusion, pathways, network):

    features = []

    features.append(np.mean(diffusion, axis=1))
    features.append(np.std(diffusion, axis=1))
    features.append(np.max(pathways, axis=1))
    features.append(network["centrality"])

    return np.vstack(features).T