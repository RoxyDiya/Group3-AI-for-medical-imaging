#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from augment import SliceAugmentor
from PIL import Image
import random, pickle

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from UNet import UNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy, DiceCE, CEclDice)

# Worker initialization function for DataLoader reproducibility
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        # multi-window slices are stored as RGB, one CT window per channel
        img = np.array(img).transpose(2, 0, 1) if img.mode == 'RGB' else np.array(img.convert('L'))[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def class_weights(dataset) -> Tensor:
    # ENet paper weighting, w = 1 / ln(1.02 + class frequency), counted on the training split only
    counts = sum(dataset.gt_transform(Image.open(gt)).sum((1, 2)) for _, gt in dataset.files)
    weights = 1 / torch.log(1.02 + counts / counts.sum())
    print(f">> Class weights {weights.tolist()}")
    return weights

def foreground_flags(dataset) -> np.ndarray:
    """True for each slice whose ground truth contains at least one organ.

    The GT PNGs hold the x63 encoding, so any non-zero pixel means an
    organ is present and no decoding is needed.
    """
    flags = np.zeros(len(dataset.files), dtype=bool)
    for i, (_, gt_path) in enumerate(dataset.files):
        flags[i] = bool(np.any(np.array(Image.open(gt_path)) > 0))
    return flags


def build_sampler(dataset, fg_ratio: float, seed: int) -> WeightedRandomSampler:
    """Draw roughly `fg_ratio` of each batch from slices containing an organ.

    Unlike dropping empty slices, the model still sees empty space, so it
    keeps learning to predict nothing there. One epoch still draws
    len(dataset) slices, so the number of gradient steps is unchanged and
    the comparison against the baseline stays fair.
    """
    fg = foreground_flags(dataset)
    n_fg, n_bg = int(fg.sum()), int((~fg).sum())
    print(f">> Sampler: {n_fg} slices with an organ, {n_bg} without; "
          f"targeting {fg_ratio:.0%} foreground per batch")

    if n_fg == 0 or n_bg == 0:
        raise ValueError("Oversampling needs both foreground and empty slices. "
                         "Was this dataset sliced with --skip_empty?")

    weights = np.where(fg, fg_ratio / n_fg, (1.0 - fg_ratio) / n_bg)

    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=g)
def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:


    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    if args.kernels:
        kernels = args.kernels
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net_cls = datasets_params[args.dataset]['net']
    if args.network == 'enet':
        net_cls = ENet
    elif args.network == 'unet':
        net_cls = UNet

    # one input channel per stacked slice (1 for 2D, 3 or 5 for 2.5D), times the
    # channels per slice image (3 for the multi-window variant, 1 otherwise)
    first_img = next(((args.data_dir or Path("data") / args.dataset) / 'train' / 'img').glob('*.png'))
    channels = len(Image.open(first_img).getbands())
    in_dim = args.context_slices * channels + (3 if args.position else 0)  # + x, y, z position channels
    net = net_cls(in_dim, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    # optimizer
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
        )

    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
        )

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=0.9,
        )   

    else: raise ValueError(f"Unknown optimizer {args.optimizer}")
    print(f">> Using {args.optimizer} optimizer with learning rate {lr}")



    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    # --data_dir lets two preprocessing variants of the same dataset live side by
    # side, so their slice directories do not overwrite each other.
    root_dir = args.data_dir if args.data_dir else Path("data") / args.dataset

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=partial(gt_transform, K),
                             augment=args.augment,
                             debug=args.debug,
                             context_slices=args.context_slices,
                             position=args.position)
    if args.sampling == "oversample":
        train_loader = DataLoader(train_set,
                                  batch_size=B,
                                  num_workers=5,
                                  sampler=build_sampler(train_set, args.fg_ratio, args.seed),
                                  worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=B,
                                  num_workers=5,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           context_slices=args.context_slices,
                           position=args.position)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            worker_init_fn=worker_init_fn)

    args.dest.mkdir(parents=True, exist_ok=True)

    # Extract and save train/val split
    train_patient_ids = sorted(set(Path(f[0]).stem.rsplit('_', 1)[0] for f in train_set.files))
    val_patient_ids = sorted(set(Path(f[0]).stem.rsplit('_', 1)[0] for f in val_set.files))
    
    split_dict = {
        'train': train_patient_ids,
        'val': val_patient_ids,
        'seed': args.seed
    }

    with open(args.dest / "train_val_split.pkl", 'wb') as f:
        pickle.dump(split_dict, f, pickle.HIGHEST_PROTOCOL)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    match args.loss:
        case 'ce':
            loss_fn = CrossEntropy(idk=idk)
        case 'wce':
            loss_fn = CrossEntropy(idk=idk, weights=class_weights(train_loader.dataset).to(device))
        case 'focal':
            loss_fn = CrossEntropy(idk=idk, gamma=2)
        case 'dicece':
            loss_fn = DiceCE(idk=idk)
        case 'cldice':
            loss_fn = CEclDice(idk=idk)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    # Per patient 3D Dice: overlap counts summed over all slices of a volume,
    # the same quantity metrics_3d.py reports, so --select 3d picks the epoch
    # on the metric we evaluate with instead of the per slice 2D Dice.
    log_dice3d_val: Tensor = torch.zeros((args.epochs, K))

    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice: float = 0

    for e in range(args.epochs):
        volumes: dict[str, Tensor] = {}
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        counts = torch.stack([einsum("bkwh->bk", pred_seg * gt),
                                              einsum("bkwh->bk", pred_seg),
                                              einsum("bkwh->bk", gt)], dim=1).float().cpu()
                        for stem, c in zip(data['stems'], counts):
                            patient = stem.rsplit('_', 1)[0]
                            volumes[patient] = volumes.get(patient, 0) + c

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        inter, pred_size, gt_size = torch.stack(list(volumes.values())).unbind(dim=1)
        dice3d = torch.where(pred_size + gt_size > 0, 2 * inter / (pred_size + gt_size), 1.0)
        log_dice3d_val[e] = dice3d.mean(dim=0)
        np.save(args.dest / "dice3d_val.npy", log_dice3d_val)

        if scheduler:
            scheduler.step()

        current_dice: float = (log_dice3d_val[e, 1:].mean().item() if args.select == '3d'
                               else log_dice_val[e, :, 1:].mean().item())
        if current_dice > best_dice:
            message = f">>> Improved {args.select} dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='ce', choices=['ce', 'wce', 'focal', 'dicece', 'cldice'],
                        help="ce is the baseline. wce weights classes by training frequency, "
                             "focal uses gamma 2, dicece adds soft Dice to ce, cldice adds a "
                             "soft centreline Dice on esophagus, trachea and aorta to ce.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument("--track_co2", action="store_true", 
                        help="Track energy use and CO2 emissions during the run.")
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--data_dir', type=Path, default=None,
                        help="Directory holding the sliced data. Defaults to data/<dataset>. "
                             "Set it to keep several preprocessing variants side by side.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed for weight init and batch shuffling. Without it, two runs "
                             "of the same config differ, and a gap between configs cannot be "
                             "told apart from run-to-run noise.")
    parser.add_argument('--sampling', default='uniform',
                        choices=['uniform', 'oversample'],
                        help="uniform = every slice equally likely (default); "
                             "oversample = draw foreground slices more often")
    parser.add_argument('--fg_ratio', type=float, default=0.66,
                        help="Target fraction of each batch containing an organ")

    parser.add_argument('--augment', action='store_true',
                        help="Random affine and elastic deformation on the "
                             "training slices, following the SegTHOR paper. "
                             "Labels are resampled nearest-neighbour.")

    parser.add_argument('--network', default=None, choices=['enet', 'unet'],
                        help="Override the dataset's default network (ENet for SEGTHOR).")
    parser.add_argument('--optimizer', default='adam', choices=['adam','adamw','sgd'],
                        help='Optimizer to use for training')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Learning rate')
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine'],
                        help="none keeps the learning rate fixed (baseline); cosine decays it "
                             "to zero over the epochs.")
    parser.add_argument('--select', default='2d', choices=['2d', '3d'],
                        help="Metric that picks the best epoch: 2d = mean per slice Dice "
                             "(baseline), 3d = mean per patient Dice, as reported.")
    parser.add_argument('--kernels', default=None, type=int,
                        help="Base channel width of the network. Defaults to the dataset "
                             "setting (8 for SEGTHOR).")
    parser.add_argument('--context_slices', default=1, type=int, choices=[1, 3, 5],
                        help="Number of neighbouring slices stacked as input channels. "
                             "1 = plain 2D (baseline), 3 or 5 = 2.5D.")
    parser.add_argument('--position', action='store_true',
                        help="Add x, y and relative slice height (z) as three extra input channels.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pprint(args)

    if args.track_co2:
        from codecarbon import OfflineEmissionsTracker, OutputMethod

        args.dest.mkdir(parents=True, exist_ok=True)
        tracker = OfflineEmissionsTracker(project_name="Medical Imaging Segmentation", country_iso_code="NLD", 
                                          output_dir=str(args.dest), output_file="emissions.csv", 
                                          output_methods=[OutputMethod.CSV], measure_power_secs=10)
        tracker.start()

        try:
            runTraining(args)
        finally:
            emissions = tracker.stop()

        if emissions is not None:
            print(f"Total CO2 emissions: {emissions:.6f} kg CO2eq")
    else:
        runTraining(args)


if __name__ == '__main__':
    main()