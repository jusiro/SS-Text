import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ss_text_solver(features, labels, model, classes, text_lambda="adaptive_shots",
                   balanced_importance=False, repel=False):

    # Number of samples
    N = labels.shape[0]

    # Compute new class centers
    with torch.no_grad():

        # Labels to ohe
        affinity_labeled = torch.nn.functional.one_hot(labels, num_classes=len(classes)).float()

        # Compute new class centers (visual)
        tau = (1 / model.adapter.logit_scale.exp().item())  # temperature scale
        vision_mu = torch.einsum('ij,ik->jk', affinity_labeled, features) / tau

        # Extract zero-shot prototypes
        text_mu = model.adapter.prototypes.data.clone().to("cpu")

        if text_lambda == "adaptive_shots":
            if balanced_importance:
                lambda_text = torch.tensor((1 / (N)))
            else:
                counts = np.array([sum(labels == i_label) for i_label in classes])
                lambda_text = torch.tensor((1 / (counts))).to(torch.float32)
        elif text_lambda == "adaptive_perf":  # CLAP - alike constraint

            # Estimate zero-shot performance
            z = torch.softmax(model(features.to(device)), -1)

            # Get one-hot encoding ground-truth
            labels_one_hot = affinity_labeled.clone()

            # Compute prior
            lambda_text = torch.diag(labels_one_hot.t().to(device) @ z.to(device))
            lambda_text /= labels_one_hot.sum(dim=0).to(device)
            lambda_text = lambda_text.clone().cpu()

            #  Correct Nans
            lambda_text[torch.isnan(lambda_text)] = torch.mean(
                lambda_text[torch.logical_not(torch.isnan(lambda_text))])

        else:  # Text lambda fixed
            lambda_text = torch.tensor(0.1)

        # Avoid NaN
        lambda_text = lambda_text.clamp(min=1e-3)

        # Compute optimum weights via ss-text solver
        new_mu = (1/N) * (1/lambda_text).unsqueeze(-1) * vision_mu + text_mu

        if repel:
            delta = (2 * new_mu - new_mu.mean(0))
            delta /= delta.norm(dim=-1, keepdim=True)
            new_mu = new_mu - delta

    # Set Adapter
    model.adapter.prototypes.data = new_mu.to(device)

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model