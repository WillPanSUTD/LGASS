import numpy as np
import random


def test_set_seed_makes_numpy_deterministic():
    from util.seeding import set_seed
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert (a == b).all()


def test_set_seed_makes_random_deterministic():
    from util.seeding import set_seed
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b
