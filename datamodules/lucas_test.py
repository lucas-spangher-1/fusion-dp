from . import lucas_processing
import pickle

def test_test_train_split_consistent():
    data = pickle.load(open('data/lucas_data_f32.pickle', 'rb'))
    train1, test1 = lucas_processing.get_train_test_indices_from_Jinxiang_cases(data, 8, "cmod", 42)
    train2, test2 = lucas_processing.get_train_test_indices_from_Jinxiang_cases(data, 8, "cmod", 42)

    assert train1 == train2
    assert test1 == test2
    assert len(set(train1)) == len(train1)
    assert len(set(test1)) == len(test1)

    assert round(float(len(test1)) / (len(train1) + len(test1)), 2) == 0.12 # around 12.5% in test/train

    train3, test3 = lucas_processing.get_train_test_indices_from_Jinxiang_cases(data, 8, "cmod", 43)
    assert train3 != train1
    assert test3 != test1