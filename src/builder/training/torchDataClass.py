"""Data export functions."""

import torch


class TrainingTensorDataset(torch.utils.data.Dataset):
    """Any number of row-aligned tensors, yielded in the order they were given.

    Replaces the TensorDatasetTwo/Three/Four/Five ladder, which had to grow a new class
    every time the pipeline gained an array -- and which encoded the arity in the class
    NAME, so adding one meant touching every construction site and every unpack site at
    once. Here the arity is data: the dataset carries an ordered list of names, and the
    training loops index (`batch[:4]`, `batch[4:6]`) rather than destructuring.

    Named access is preserved, so `ds.features`, `ds.molelabels` and friends keep working
    for anything that reaches past `__getitem__`.

    The row-count assertion names the offending array. A silent mismatch here is the worst
    kind of bug in this pipeline: derivative rows paired with the wrong composition still
    train, still converge, and are wrong.
    """

    def __init__(self, **tensors):
        items = [(k, v) for k, v in tensors.items() if v is not None]
        if not items:
            raise ValueError("TrainingTensorDataset needs at least one tensor")
        self.names = [k for k, _ in items]
        n = items[0][1].shape[0]
        for name, t in items:
            if t.shape[0] != n:
                raise AssertionError(
                    f"'{name}' has {t.shape[0]} rows against '{self.names[0]}''s {n}; "
                    f"every array in a bundle must be row-parallel.")
            setattr(self, name, t)

    def __len__(self):
        return getattr(self, self.names[0]).shape[0]

    def __getitem__(self, idx):
        return tuple(getattr(self, n)[idx] for n in self.names)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]


class TensorDatasetThree(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, binarylabels: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        assert features.shape[0] == binarylabels.shape[0], "Features and binary labels must have same length"

        self.binarylabels = binarylabels
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
            return self.features[idx], self.binarylabels[idx], self.labels[idx]


class TensorDatasetFour(TrainingTensorDataset):
    """The historical 4-tuple, kept so existing call sites are untouched.

    Optional `dndp`/`dndt` append to the tuple, which is what derivative-supervised
    training consumes: `trainer.py` reads `batch[:4]` and ignores anything past it, and
    `sobolev.py` reads `batch[4:6]`. So one dataset serves both loops and a bundle without
    derivatives yields exactly the 4-tuple it always did.
    """

    def __init__(self, features, binarylabels, labels, molelabels, dndp=None, dndt=None):
        if (dndp is None) != (dndt is None):
            raise ValueError(
                "dndp and dndt must be supplied together -- one without the other means "
                "half the derivative supervision is silently missing.")
        super().__init__(features=features, binarylabels=binarylabels, labels=labels,
                         molelabels=molelabels, dndp=dndp, dndt=dndt)

    @property
    def has_derivatives(self):
        return 'dndp' in self.names and 'dndt' in self.names


class TensorDatasetFive(TrainingTensorDataset):
    """Four plus `freeoutputs`.

    NOTE: the previous implementation's `__getitem__` returned only the first four
    tensors, so `freeoutputs` was stored and never yielded -- this class behaved exactly
    like TensorDatasetFour. Nothing depended on that (free_outputs is never loaded, see
    loadTrainData), but the tuple now includes it, which is what the name promises.

    Deliberately does NOT take derivatives. The training loops locate them positionally at
    `batch[4:6]`, and here index 4 is `freeoutputs` -- so combining the two would hand the
    derivative loss the wrong tensor with matching-enough shapes to go unnoticed. If
    free_outputs ever does become a training input, give the loops a named accessor first.
    """

    def __init__(self, features, binarylabels, labels, molelabels, freeoutputs):
        super().__init__(features=features, binarylabels=binarylabels, labels=labels,
                         molelabels=molelabels, freeoutputs=freeoutputs)
