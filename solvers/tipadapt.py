import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def tipadapt_solver(features, labels, model, classes):

    # l2-norm features
    features /= features.norm(dim=-1, keepdim=True)

    # Set key and values
    model.adapter.cache_keys = torch.transpose(features, 1, 0).to(torch.float32).to(device)
    model.adapter.cache_values = torch.nn.functional.one_hot(labels, num_classes=len(classes)).to(torch.float32).to(device)

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model