import torch.nn.functional as F
from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.weights = kwargs.get('weights')
        self.gamma = kwargs.get('gamma', 0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        log_p = (p + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()
        if self.weights is not None:
            mask = mask * self.weights[self.idk, None, None]
        norm = mask.sum() + 1e-10
        if self.gamma:
            mask = mask * (1 - p) ** self.gamma

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= norm

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        inter = einsum("bkwh,bkwh->k", p, g)
        total = einsum("bkwh->k", p) + einsum("bkwh->k", g)
        dice = (2 * inter + 1) / (total + 1)

        return 1 - dice.mean()


class DiceCE():
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(**kwargs)
        self.dice = DiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        return self.ce(pred_softmax, weak_target) + self.dice(pred_softmax, weak_target)


def soft_erode(x):
    return -F.max_pool2d(-x, 3, stride=1, padding=1)


def soft_skeleton(x, iters):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021): peel the shape
    with min pooling and keep what an opening would remove at each step."""
    skel = F.relu(x - F.max_pool2d(soft_erode(x), 3, stride=1, padding=1))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - F.max_pool2d(soft_erode(x), 3, stride=1, padding=1))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CEclDice():
    """Cross-entropy plus a soft clDice term on the tube shaped organs.

    Our error analysis showed the lower esophagus is mostly missed: the
    prediction breaks the tube into pieces. Cross-entropy charges a gap in
    the middle of a tube the same as a pixel at its edge; clDice compares
    centrelines, so a gap costs the whole stretch of centreline it cuts.
    Applied to esophagus (1), trachea (3) and aorta (4), not the heart.
    """
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(idk=kwargs['idk'])
        self.tubes = kwargs.get('tubes', [1, 3, 4])
        self.weight = kwargs.get('weight', 0.5)
        self.iters = kwargs.get('iters', 10)  # above the aorta's radius in pixels at 256 x 256
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        p = pred_softmax[:, self.tubes, ...]
        g = weak_target[:, self.tubes, ...].float()
        skel_p, skel_g = soft_skeleton(p, self.iters), soft_skeleton(g, self.iters)

        # per class over the batch; +1 keeps a class absent from the batch at no cost
        tprec = (einsum("bkwh->k", skel_p * g) + 1) / (einsum("bkwh->k", skel_p) + 1)
        tsens = (einsum("bkwh->k", skel_g * p) + 1) / (einsum("bkwh->k", skel_g) + 1)
        cldice = 2 * tprec * tsens / (tprec + tsens)

        return self.ce(pred_softmax, weak_target) + self.weight * (1 - cldice.mean())
