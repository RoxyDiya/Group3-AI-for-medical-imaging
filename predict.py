import argparse
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SliceDataset
from main import img_transform, gt_transform
from utils import probs2class, save_images, tqdm_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help="bestmodel.pkl of a run")
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--context_slices', default=1, type=int, choices=[1, 3, 5])
    parser.add_argument('--position', action='store_true',
                        help="The model was trained with --position")
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    net = torch.load(args.model, map_location=device, weights_only=False)
    net.eval()

    K = 5
    data = SliceDataset(args.subset, args.data_dir,
                        img_transform=img_transform,
                        gt_transform=partial(gt_transform, K),
                        context_slices=args.context_slices,
                        position=args.position)
    loader = DataLoader(data, batch_size=8, num_workers=4, shuffle=False)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for batch in tqdm_(loader, desc=f">> Predicting {args.subset}"):
            probs = F.softmax(net(batch['images'].to(device)), dim=1)
            save_images(probs2class(probs) * 63, batch['stems'], args.dest)

    print(f"Saved {len(data)} predictions to {args.dest}")


if __name__ == "__main__":
    main()
