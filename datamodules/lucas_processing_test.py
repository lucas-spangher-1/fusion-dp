from . import lucas_processing
import pickle


def get_data():
    if not get_data.data:
        get_data.data = pickle.load(open("data/lucas_data_f32.pickle", "rb"))
    return get_data.data


# hack to load data local to the method
get_data.data = None


def test_consistency_in_rand_and_overlap():
    data = get_data()
    for case in range(1, 12 + 1):
        print(case)
        (
            train_inds,
            test_inds,
        ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
            data, case, "east", 42, test_percentage=0.15
        )
        assert set(train_inds) & set(test_inds) == set(), "train and test overlap"
        (
            train_inds2,
            test_inds2,
        ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
            data, case, "east", 42, test_percentage=0.15  # should be the same
        )
        assert train_inds == train_inds2
        assert test_inds == test_inds2
        (
            train_inds3,
            test_inds3,
        ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
            data, case, "east", 85, test_percentage=0.15  # new seed!
        )
        assert train_inds != train_inds3
        assert test_inds != test_inds3
        assert set(test_inds) != set(test_inds3)
        if case != 6:  # case 6 has no new machine, so train isn't affected
            assert set(train_inds) != set(train_inds3)


def test_test_percentage():
    data = get_data()
    existing, new, disr, nondisr = lucas_processing.get_index_sets(
        data, data.keys(), "east"
    )
    for p in [0.15, 0.30, 0.45]:
        (
            _,
            test_inds,
        ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
            data, 8, "east", 42, test_percentage=p
        )
        assert appr(set(test_inds) & new, new) == p, "test is 15pct of new"


def appr(a, b):
    """Approximate the size of a relative to b to 2 decimal places

    Args:
        a: collection
        b: collection

    Returns:
        (float): percentage of N
    """
    return round(len(a) / len(b), 2)


def case_setup(case_number):
    data = get_data()
    # check consistency when called twice
    (
        train_inds,
        test_inds,
    ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
        data, case_number, "east", 42, test_percentage=0.15
    )
    train_inds, test_inds = set(train_inds), set(test_inds)
    # break down everything except for test indices
    ix = lucas_processing.get_index_sets(data, set(data.keys()) - test_inds, "east")
    # breakdown within training inds
    tix = lucas_processing.get_index_sets(data, train_inds, "east")
    return ix, tix


def test_case_1():
    ix, tix = case_setup(1)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert len(tix.new & tix.disr) == 20


def test_case_2():
    ix, tix = case_setup(2)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert (tix.new & tix.disr) == set()


def test_case_3():
    ix, tix = case_setup(3)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert appr(tix.new & tix.nondisr, ix.new & ix.nondisr) == 0.5
    assert len(tix.new & tix.disr) == 20


def test_case_4():
    ix, tix = case_setup(4)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == set()
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert len(tix.new & tix.disr) == 20


def test_case_5():
    ix, tix = case_setup(5)
    assert (tix.existing & tix.nondisr) == (ix.existing & ix.nondisr)
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert len(tix.new & tix.disr) == 20


def test_case_6():
    ix, tix = case_setup(6)
    assert (tix.existing & tix.nondisr) == (ix.existing & ix.nondisr)
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == set()
    assert (tix.new & tix.disr) == set()


def test_case_7():
    ix, tix = case_setup(7)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert (tix.new & tix.disr) == (ix.new & ix.disr)


def test_case_8():
    ix, tix = case_setup(8)
    assert (tix.existing & tix.nondisr) == (ix.existing & ix.nondisr)
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert (tix.new & tix.disr) == (ix.new & ix.disr)


def test_case_9():
    ix, tix = case_setup(9)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == set()
    assert (tix.new & tix.nondisr) == (ix.new & ix.nondisr)
    assert (tix.new & tix.disr) == (ix.new & ix.disr)


def test_case_10():
    ix, tix = case_setup(10)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == (ix.existing & ix.disr)
    assert appr(tix.new & tix.nondisr, ix.new & ix.nondisr) == 0.33
    assert (tix.new & tix.disr) == (ix.new & ix.disr)


def test_case_11():
    ix, tix = case_setup(11)
    assert appr(tix.existing & tix.nondisr, ix.existing & ix.nondisr) == 0.20
    assert (tix.existing & tix.disr) == set()
    assert appr(tix.new & tix.nondisr, ix.new & ix.nondisr) == 0.33
    assert (tix.new & tix.disr) == (ix.new & ix.disr)


def test_case_12():
    ix, tix = case_setup(12)
    assert (tix.existing & tix.nondisr) == set()
    assert (tix.existing & tix.disr) == set()
    assert appr(tix.new & tix.nondisr, ix.new & ix.nondisr) == 0.33
    assert (tix.new & tix.disr) == (ix.new & ix.disr)
