from abc import abstractmethod, ABC
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.random import SeedSequence, MT19937, RandomState


class TrainValTestSpliter(ABC):
    @abstractmethod
    def split(self, instances: np.ndarray) -> Tuple[Sequence, Sequence, Sequence]:
        raise NotImplementedError()

    @abstractmethod
    def is_reproducible(self) -> bool:
        raise NotImplementedError()


class OrderedSplitter(TrainValTestSpliter):
    def __init__(self, train: float = 0.6, val: float = 0.2, reverse: bool = False) -> None:
        assert train + val <= 1.0
        self._train = train
        self._val = val
        self._reverse = reverse

    def split(self, instances: Sequence[np.ndarray]) \
            -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
        n_instances = len(instances)
        n_train = int(n_instances * self._train)
        n_val = int(n_instances * self._val)
        n_test = n_instances - n_train - n_val
        if self._reverse:
            test_instances = instances[:n_test]
            val_instances = instances[n_test:n_test + n_val]
            train_instances = instances[n_test + n_val:]
        else:
            train_instances = instances[:n_train]
            val_instances = instances[n_train:n_train + n_val]
            test_instances = instances[n_train + n_val:]
        return train_instances, val_instances, test_instances

    def is_reproducible(self) -> bool:
        return True


class CustomOrderSplitter(TrainValTestSpliter):
    def __init__(self, order_idxes: np.ndarray, train: float = 0.6, val: float = 0.2) -> None:
        assert train + val <= 1.0
        self._train = train
        self._val = val
        self._order_idxes = order_idxes
        assert order_idxes.ndim == 1
        self._n_instances = len(self._order_idxes)

    def split(self, instances: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_instances = self._n_instances
        assert n_instances == len(instances)
        sorted_idxes = self._order_idxes
        n_train = int(n_instances * self._train)
        n_val = int(n_instances * self._val)
        train_instances = instances[sorted_idxes[:n_train]]
        val_instances = instances[sorted_idxes[n_train:n_train + n_val]]
        test_instances = instances[sorted_idxes[n_train + n_val:]]
        return train_instances, val_instances, test_instances

    def is_reproducible(self) -> bool:
        return True


class ChronoRatioSpliter(CustomOrderSplitter):

    def __init__(self, time_values: Optional[np.ndarray], train: float = 0.6, val: float = 0.2) -> None:
        assert train + val <= 1.0
        self._train = train
        self._val = val
        self._sorted_idxes = np.argsort(time_values)
        assert time_values.ndim == 1
        super().__init__(self._sorted_idxes, train, val)

    def is_reproducible(self) -> bool:
        return True


class RandomSpliter(TrainValTestSpliter):

    def __init__(self,
                 train: float = 0.6, val: float = 0.2,
                 random_state: Optional[int] = None
                 ) -> None:
        assert train + val <= 1.0
        self._train = train
        self._val = val
        self._random_state = random_state

    def split(self, instances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Next random is thread-safe and can be used for reproducibility
        n_instances = len(instances)
        idx = np.arange(n_instances)
        rs = RandomState(MT19937(SeedSequence(self._random_state)))
        rs.shuffle(idx)
        n_train = int(n_instances * self._train)
        n_val = int(n_instances * self._val)
        train_instances = instances[idx[:n_train]]
        val_instances = instances[idx[n_train:n_train + n_val]]
        test_instances = instances[idx[n_train + n_val:]]
        return train_instances, val_instances, test_instances

    def is_reproducible(self) -> bool:
        return self._random_state is not None
