import torch
import numpy as np

from solvers.gd import gd_solver
from solvers.ss import ss_solver
from solvers.tipadapt import tipadapt_solver
from solvers.sstext import ss_text_solver
from solvers.lpp2 import lp_pp_solver

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_adaptation(features, labels, model, classes, solver="SStext", samples_text=None, labels_text=None):

    # Extended cross-modal dataset
    if solver == "CrossModal":
        if samples_text is not None:
            features = torch.concatenate([features, samples_text], dim=0)
            labels = torch.concatenate([labels, labels_text], dim=0)

    if solver in ["LP", "LPcw", "LPldam", "TaskRes", "ClipAdapt", "CrossModal", "CLAP"]:
        z, model = gd_solver(features, labels, model, classes=classes,
                             weights=("cw" in solver or "ldam" in solver),
                             ldam=("ldam" in solver),
                             clap=(solver == "CLAP"))
    elif solver == "SS":
        z, model = ss_solver(features, labels, model)
    elif solver == "SStext":
        z, model = ss_text_solver(features, labels, model, classes=classes,
                                  balanced_importance=True, repel=False)
    elif solver == "SStext+":
        z, model = ss_text_solver(features, labels, model, classes=classes,
                                  balanced_importance=False, repel=True)
    elif solver == "LP++":
        z, model = lp_pp_solver(features, labels, model, classes=classes)
    elif solver == "LP++(TF)":
        z, model = lp_pp_solver(features, labels, model, classes=classes, epochs=0)
    elif solver == "TIPAd":
        z, model = tipadapt_solver(features, labels, model, classes=classes)
    elif solver == "TIPAdFT":
        _, model = tipadapt_solver(features, labels, model, classes=classes)
        model.adapter.cache_values = torch.nn.Parameter(model.adapter.cache_values)
        z, model = gd_solver(features, labels, model, classes=classes)
    elif solver == "zero-shot":
        with torch.no_grad():
            z = torch.softmax(model(features.to(device)), -1)
    else:
        print("Solver not implemented...")
        return None, model

    return z, model
