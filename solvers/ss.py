import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ss_solver(features, labels, model):

    # Number of samples
    N = labels.shape[0]

    # Compute new class centers
    with torch.no_grad():

        # Labels to ohe
        affinity_labeled = torch.nn.functional.one_hot(labels).float()

        # Compute new class centers (visual)
        vision_mu = torch.einsum('ij,ik->jk', affinity_labeled, features)

        # Normalize
        vision_mu /= affinity_labeled.sum(0).unsqueeze(-1)

    # Set Adapter
    model.adapter.prototypes.data = vision_mu.to(device)

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model