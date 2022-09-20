#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from unittest import TestCase, main

import numpy as np

from keras_generators.splitters import ChronoRatioSpliter, RandomSpliter


class TestChronoRatioSpliter(TestCase):
    def test_split(self):
        time_values = (np.arange(10) * 2)[::-1]
        splitter = ChronoRatioSpliter(time_values)
        na = np.arange(10)
        train, val, test = splitter.split(na)
        self.assertTrue(np.array_equal(train, np.array([9, 8, 7, 6, 5, 4])))
        self.assertTrue(np.array_equal(val, np.array([3, 2])))
        self.assertTrue(np.array_equal(test, np.array([1, 0])))


class TestRandomSpliter(TestCase):
    def test_split_reproductibility(self):
        na = np.arange(100)
        splitter = RandomSpliter(random_state=42)
        train, val, test = splitter.split(na)
        trainr, valr, testr = splitter.split(na)
        self.assertTrue(np.array_equal(train, trainr))
        self.assertTrue(np.array_equal(val, valr))
        self.assertTrue(np.array_equal(test, testr))
        splitter2 = RandomSpliter(random_state=42)
        train2, val2, test2 = splitter2.split(na)
        self.assertTrue(np.array_equal(train, train2))
        self.assertTrue(np.array_equal(val, val2))
        self.assertTrue(np.array_equal(test, test2))
        # and the next one should be distinct by using different random state
        splitter3 = RandomSpliter(random_state=0)
        train3, val3, test3 = splitter3.split(na)
        self.assertFalse(np.array_equal(train, train3))
        self.assertFalse(np.array_equal(val, val3))
        self.assertFalse(np.array_equal(test, test3))


if __name__ == '__main__':
    main()
