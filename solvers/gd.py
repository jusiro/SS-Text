import torch
import random
import sklearn

import numpy as np

from solvers.utils import LDAMLoss

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gd_solver(features, labels, model, classes, weights=False, ldam=False, clap=False,
              base_lr=0.1, epochs=300, bs=5000, disp=False):

    # Define constants
    N = features.shape[-1]

    # Prepare loss for ldam
    if ldam:
        cweight = sklearn.utils.class_weight.compute_class_weight("balanced", classes=classes,
                                                                  y=labels.cpu().numpy())
        counts = np.array([sum(labels.cpu().numpy() == i_label) for i_label in classes])
        LDAM = LDAMLoss(cls_num_list=counts, weight=torch.tensor(cweight).to(device).float())

    # Move input to gpu
    features, labels, model = features.to(device), labels.to(device), model.to(device)

    # Set training optimizer
    optim = torch.optim.SGD(params=model.parameters(), lr=base_lr, weight_decay=0.0, momentum=0.9)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs)

    # Set constraint
    if clap:
        with torch.no_grad():
            # Compute zero-shot predictions
            z = torch.softmax(model(features.to(device)), -1)

            # Get one-hot encoding ground-truth
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(classes)).float().clone()

            # Compute prior
            lambda_text = torch.diag(labels_one_hot.t().to(device) @ z.to(device))
            lambda_text /= labels_one_hot.sum(dim=0).to(device)
            lambda_text = lambda_text.clone().unsqueeze(-1).cpu()

            #  Correct Nans
            lambda_text[torch.isnan(lambda_text)] = torch.mean(lambda_text[torch.logical_not(torch.isnan(lambda_text))])

            # Compute zero-shot prototypes
            zero_shot_prot = model.adapter.prototypes.data.clone()

    # Fit Adapter
    for i_epoch in range(epochs):

        # Set training indexes
        indexes = np.arange(0, features.shape[0])
        random.shuffle(indexes)
        tracking_loss = 0.0

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
            if weights:
                if ldam:
                    loss = LDAM(logits, y)
                else:
                    cweight = sklearn.utils.class_weight.compute_class_weight("balanced", classes=classes,
                                                                              y=labels.cpu().numpy())
                    loss = torch.nn.functional.cross_entropy(logits, y, weight=torch.tensor(cweight).to(device).float())
            else:
                loss = torch.nn.functional.cross_entropy(logits, y)

            if clap:
                # Compute prototype deviation
                disimilitude = (model.adapter.prototypes - zero_shot_prot.clone()).pow(2).sum(-1)

                # Compute weighted penalty
                loss_clap = torch.mean(lambda_text.to(device) * disimilitude)

                # Overall loss
                loss += loss_clap

            # Update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Update scheduler
            scheduler.step()

            # Update tracking loss
            tracking_loss += loss.item() / (max(1, N // bs))

            if disp:
                print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                    i_epoch=i_epoch + 1, epochs=epochs, i_step=i_step + 1, steps=int(N // bs),
                    loss=round(loss.item(), 4)), end="\r")
        if disp:
            print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                i_epoch=i_epoch + 1, epochs=epochs, i_step=int(N // bs),
                steps=int(N // bs), loss=round(tracking_loss, 4)), end="\n")

    model.eval()

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features), -1)

    return z, model
