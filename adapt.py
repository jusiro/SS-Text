"""
Main function for few-shot adaptation.
"""

import argparse
import torch
import os
import time

import pandas as pd

from tqdm.auto import tqdm

from modeling.adapters.models import Adapter
from solvers.runner import run_adaptation
from datetime import datetime

from data.configs import get_task_setting, get_experiment_setting
from data.utils import *
from utils.metrics import *
from local_data.constants import *
from modeling.vlms.constants import *
from data.constants import *

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # Prepare table to save results
    res = pd.DataFrame()

    for i_task in range(0, len(args.tasks)):
        print("  Testing on: [{dataset}]".format(dataset=args.tasks[i_task]))

        # Identify task
        args.task = args.tasks[i_task]

        # Get vlm id
        if args.vlm_id is None:
            vlm_id = task_to_vlm[args.tasks[i_task]]
        else:
            vlm_id = args.vlm_id

        # Retrieve task details (i.e. experiment for ID, experiments for OOD, ...)
        get_task_setting(args)

        # ----------------------------------------
        # Load adaptation data
        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        setting = get_experiment_setting(args.task_setting["experiment"])
        print("  Adapting on: [{dataset}]".format(dataset=setting["experiment"]))
        # Load data
        id = "./local_data/cache/" + setting["experiment"] + "_" + vlm_id.lower().replace("/", "_")
        if os.path.isfile(id + ".npz"):
            print("  Loading features from cache_features")
            cache_adapt = np.load(id + ".npz", allow_pickle=True)
        else:
            print("Training data not found... return")
            return

        # ----------------------------------------
        # Load testing data
        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        setting = get_experiment_setting(args.task_setting["experiment_test"][0])
        print("  Adapting on: [{dataset}]".format(dataset=args.task_setting["experiment_test"][0]))
        # Load data
        id = "./local_data/cache/" + setting["experiment"] + "_" + vlm_id.lower().replace("/", "_")
        if os.path.isfile(id + ".npz"):
            print("  Loading features from cache_features")
            cache_test = np.load(id + ".npz", allow_pickle=True)
        else:
            print("Training data not found... return")
            return

        # Run for different seeds
        top1, bal_acu, time_adapt = [], [], []
        for _ in tqdm(range(args.seeds), leave=False, desc="  Few-shot adaptation: "):
            torch.cuda.empty_cache()

            # Get test logits and labels
            feats_adapt, labels_adapt = cache_adapt["feats_ds"], np.int8(cache_adapt["refs_ds"])
            feats_test, labels_test = cache_test["feats_ds"], np.int8(cache_test["refs_ds"])

            # Get label-marginal distribution from testing
            if scenarios_setups[args.scenario]["split_balanced"]:
                counts = np.ones((len(np.unique(labels_test))))
            else:
                counts = np.bincount(labels_adapt)
            label_dist = counts / np.sum(counts)

            # Calibration split: retrieving few-shots from training partitions
            feats_adapt, labels_adapt = balance_split(
                feats_adapt, labels_adapt, k=args.k, p=label_dist, seed=_,
                allow_missing_class=scenarios_setups[args.scenario]["allow_missing_class"]
            )

            # Move data to gpu
            feats_test, labels_test = torch.tensor(feats_test), torch.tensor(labels_test).to(torch.long)
            feats_adapt, labels_adapt = torch.tensor(feats_adapt), torch.tensor(labels_adapt).to(torch.long)

            # Adaptation time
            time_adapt_i_1 = time.time()

            # Set Adapter
            adapter = Adapter(torch.tensor(cache_adapt["initial_prototypes"]),
                              cache_adapt["logit_scale"], adapter=args.adapt)

            # Adjust Adapter based on adaptation data
            preds_adapt, adapter = run_adaptation(feats_adapt, labels_adapt, adapter, solver=args.adapt,
                                                  classes=np.unique(labels_test.cpu().numpy()),
                                                  samples_text=torch.tensor(cache_adapt["samples_text"]),
                                                  labels_text=torch.tensor(cache_adapt["labels_text"]).to(torch.long)
                                                  )

            # Predict on test
            with torch.no_grad():
                preds_test = torch.softmax(adapter(feats_test.to(device)), -1)

            # Set transfer learning times
            time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1

            #  Run metrics
            metrics_accuracy = accuracy(preds_test.cpu(), labels_test, (1,))
            metrics_aca = aca(preds_test.cpu().numpy(), labels_test.numpy())
            # Training times
            time_adapt.append(time_adapt_i)
            # Allocate accuracy-related metrics
            top1.append(metrics_accuracy[0].item()), bal_acu.append(metrics_aca)

            # Output metrics
            print('  [AVG] ACC: [{acc}] -- ACA: [{aca}]'.format(
                acc=np.round(np.mean(top1[-1]), 3),
                aca=np.round(np.mean(bal_acu[-1]), 3)
            ))

        # Output metrics
        print("  " + "%" * 100)
        print('  [AVG] ACC: [{acc}] -- ACA: [{aca}]'.format(
            acc=np.round(np.mean(top1), 3),
            aca=np.round(np.mean(bal_acu), 3)
        ))
        print("  " + "%" * 100)

        # Prepare results
        res_i = {"backbone": vlm_id, "dataset": args.tasks[i_task], "adapt": args.adapt,
                 "shots": args.k, "scenario": str(args.scenario),
                 "top1": np.round(np.mean(top1), 3), "aca": np.round(np.mean(bal_acu), 3),
                 "time_adapt": np.round(np.mean(time_adapt), 6)}
        res = pd.concat([res, pd.DataFrame(res_i, index=[0])])

    # Produce average results
    avg = res[["top1", "aca", "time_adapt"]].mean().values
    res_avg = {"backbone": "AVG", "dataset": "AVG", "adapt": args.adapt, "shots": args.k,
               "scenario": str(args.scenario), "top1": np.round(avg[0], 3), "aca": np.round(avg[1], 3),
               "time_adapt": np.round(avg[2], 6)}
    res = pd.concat([res, pd.DataFrame(res_avg, index=[0])])

    # save results
    path = "./local_data/results/scenario_{scenario}/".format(scenario=str(args.scenario))
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame.to_excel(res, path + str(args.k) + "shot__" + args.adapt + datetime.now().strftime("__%m-%d_%H-%M-%S") + ".xlsx")


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')

    # Tasks
    parser.add_argument('--tasks',
                        default='Gleason,Skin,NCT,MESSIDOR,MMAC,FIVES,CheXpert5x200,NIH,COVID',
                        help='Gleason,Skin,NCT,MESSIDOR,MMAC,FIVES,CheXpert5x200,NIH,COVID',
                        type=lambda s: [item for item in s.split(',')])

    # Model to employ
    parser.add_argument('--vlm_id', default=None, help='"conch", "flair", "convirt"')

    # Setting for adaptation
    parser.add_argument('--adapt', default='SStext+', help='TL mode',
                        choices=["zero-shot", "LP", "LPcw", "LPldam", "CLAP", "TaskRes", "LP++", "LP++(TF)", "TIPAd",
                                 "TIPAdFT", "ClipAdapt", "CrossModal", "SS", "SStext", "SStext+"])

    # Few-shot sampling hyperparameters
    parser.add_argument('--scenario', default='realistic', help='few-shot adaptation scenario',
                        choices=["standard", "relaxed", "realistic"])
    parser.add_argument('--k', default=1, help='Number of shots for adaptation per class', type=int)

    # Number of seeds
    parser.add_argument('--seeds', default=20, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()
