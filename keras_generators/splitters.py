from abc import abstractmethod, ABC
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.random import SeedSequence, MT19937, RandomState


class TrainValTestSpliter(ABC):
    @abstractmethod
    def split(self, instances: np.ndarray) -> Tuple[Sequence, Sequence, Sequence]:
        raise NotImplementedError()


class ChronoRatioSpliter(TrainValTestSpliter):

    def __init__(self, time_values: np.ndarray, train: float = 0.6, val: float = 0.2) -> None:
        assert train + val <= 1.0
        self._train = train
        self._val = val
        assert time_values.ndim == 1
        self._time_values = time_values
        self._sorted_idxes = np.argsort(self._time_values)
        self._n_instances = len(self._sorted_idxes)

    def split(self, instances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_train = int(self._n_instances * self._train)
        n_val = int(self._n_instances * self._val)
        train_instances = instances[self._sorted_idxes[:n_train]]
        val_instances = instances[self._sorted_idxes[n_train:n_train + n_val]]
        test_instances = instances[self._sorted_idxes[n_train + n_val:]]
        # sorted_instances = sorted(instances, key=self._key, reverse=False)
        # return sorted_instances[:n_train], sorted_instances[n_train:n_train + n_val], sorted_instances[n_train + n_val:]
        return train_instances, val_instances, test_instances


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
