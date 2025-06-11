import torch
import random

from scipy.linalg import eigh

import numpy as np


# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lp_pp_solver(features, labels, model, classes, epochs=300, bs=5000, disp=False):

    def get_one_hot(y_s: torch.tensor, num_classes: int):
        """
            args:
                y_s : torch.Tensor of shape [n_task, shot]
            returns
                y_s : torch.Tensor of shape [n_task, shot, num_classes]
        """
        one_hot_size = list(y_s.size()) + [num_classes]
        one_hot = torch.zeros(one_hot_size, device=y_s.device, dtype=torch.float16)

        one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
        return one_hot

    def compute_centroids_alpha(z_s: torch.tensor, y_s: torch.tensor, classes):
        """
        inputs:
            z_s : torch.Tensor of size [batch_size, s_shot, d]
            y_s : torch.Tensor of size [batch_size, s_shot]

        updates :
            centroids : torch.Tensor of size [n_task, num_class, d]
        """
        one_hot = get_one_hot(y_s, num_classes=len(classes))
        centroids = (one_hot * z_s / torch.clamp(one_hot.sum(-2, keepdim=True), min=1)).sum(1)  # [batch, K, d]
        return centroids

    def compute_centroids(z_s: torch.tensor, y_s: torch.tensor, classes):
        """
        inputs:
            z_s : torch.Tensor of size [batch_size, s_shot, d]
            y_s : torch.Tensor of size [batch_size, s_shot]

        updates :
            centroids : torch.Tensor of size [n_task, num_class, d]
        """
        one_hot = get_one_hot(y_s, num_classes=len(classes))
        # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
        centroids = one_hot.transpose(1, 2).to(torch.float32).bmm(z_s)  # [batch, K, d]
        return centroids

    def calculate_lr_alpha(features, clip_weights):
        # lr_alpha
        ftT = features @ clip_weights
        temp = torch.sum(torch.pow(ftT, 2), dim=0)
        max_sum = max(temp)
        lr_alpha = features.shape[0] / (max_sum * 4)
        return lr_alpha

    def calculate_init_alpha(features, labels, shots, clip_weights, classes):
        # init_alpha
        alpha_tilde = compute_centroids_alpha((features @ clip_weights).unsqueeze(0), labels.unsqueeze(0),
                                              classes)[0]
        alpha_tilde = alpha_tilde.double() * shots
        alpha_init = 250 / shots * alpha_tilde
        final_init_alpha_mean = torch.mean(alpha_init)
        return final_init_alpha_mean

    def calculate_lr_w(features):
        # lr_w
        ff_t = features.T @ features
        ff_t_np = ff_t.cpu().numpy()
        w, v = eigh(ff_t_np)
        max_eigen = max(w)  # check the iters of power iteration
        lr_w = (4 * features.shape[0]) / max_eigen
        return lr_w

    # Number of samples
    N = labels.shape[0]

    # Move input to gpu
    features, labels, model = features.to(device), labels.to(device), model.to(device)

    # L2-norm features
    features_norm = features / features.norm(dim=-1, keepdim=True)
    centroids = compute_centroids(features_norm.unsqueeze(0), labels.unsqueeze(0).to(torch.int64), classes)

    # Set weights in adapter (init weights)
    model.adapter.classifier.weight.data = centroids[0]

    # Search learning rate for visual classifier
    lr_temp = calculate_lr_w(features_norm)

    # init_alpha for blending hyper-parameter
    shots = features_norm.shape[0] / model.adapter.zero_shot_prot.shape[0]  # Number of shots
    text_prototypes = model.adapter.zero_shot_prot.clone()
    text_prototypes /= text_prototypes.norm(dim=-1, keepdim=True)
    final_init_alpha_mean = calculate_init_alpha(features, labels, shots, text_prototypes.t(), classes)
    model.adapter.alpha_vec = torch.autograd.Variable(
        final_init_alpha_mean * torch.ones(1, int(features.shape[0] / shots)).to(device),
        requires_grad=True)

    # lr_alpha
    lr_alpha = calculate_lr_alpha(features, text_prototypes.t())

    # Set optimizer
    optim = torch.optim.SGD(model.adapter.classifier.parameters(), lr_temp, momentum=0.9)

    # Move to device
    model = model.to(device)
    features = features.to(device)

    # Fit Adapter
    tracking_loss = 0.0
    for i_epoch in range(epochs):

        # Set training indexes
        indexes = np.arange(0, features.shape[0])
        random.shuffle(indexes)

        for i_step in range(max(1, N // bs)):

            # Select batch indexes
            init = int(i_step * bs)
            end = int((1 + i_step) * bs)

            # Retrieve features
            x = features[indexes[init:end], :].to(torch.float32)

            # Retrieve codes
            y = labels[indexes[init:end]].to(torch.long)

            # Forward
            logits = model.forward(x)

            # Adapter loss
            loss = torch.nn.functional.cross_entropy(logits, y)

            # Update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Update tracking loss
            tracking_loss += loss.item() / (max(1, N // bs))

            if disp:
                print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                    i_epoch=i_epoch + 1, epochs=epochs, i_step=i_step + 1, steps=int(N // bs),
                    loss=round(loss.item(), 4)), end="\r")

        # # update for alpha
        if (i_epoch + 1) % 10 == 0:
            model.adapter.alpha_vec.data -= lr_alpha * model.adapter.alpha_vec.grad.data

        if disp:
            print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                i_epoch=i_epoch + 1, epochs=epochs, i_step=int(N // bs),
                steps=int(N // bs), loss=round(tracking_loss, 4)), end="\n")

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model
