import pandas as pd

def _append_same_diff(same, diff):
    """
    Vertically Appends two dataframes passed as arguments

    :param same: same pair dataframe
    :param diff: diff pair dataframe
    :return: vertically appended dataframe
    """
    return pd.concat([same, diff])



def _drop_columns(_dataframe):
    for col in _dataframe.columns:
        if 'Unnamed' in col or 'img_id' in col:
            _dataframe.drop(col, axis=1, inplace=True)
    return _dataframe


def _concat_features(ap_data, feat_data):
    merge_matrix = pd.merge(ap_data, feat_data, left_on='img_id_A', right_on='img_id', how='left')
    merge_matrix = pd.merge(merge_matrix, feat_data, left_on='img_id_B', right_on='img_id', how='left', suffixes=('_a', '_b'))


    feature_m = _drop_columns(merge_matrix)
    target_v = feature_m['target']
    feature_m.drop('target', axis=1, inplace=True)
    return feature_m, target_v

def _subtract_features(ap_data, feat_data):
    merge_matrix_1 = pd.merge(ap_data, feat_data, left_on='img_id_A', right_on='img_id', how='left')
    merge_matrix_2 = pd.merge(ap_data, feat_data, left_on='img_id_B', right_on='img_id', how='left')
    merge_matrix_1 = _drop_columns(merge_matrix_1)
    merge_matrix_2 = _drop_columns(merge_matrix_2)

    target_v = merge_matrix_1.iloc[:,0]
    merge_matrix_1.drop('target', axis=1, inplace=True)
    merge_matrix_2.drop('target', axis=1, inplace=True)

    merge_matrix_1 = abs(merge_matrix_1.subtract(merge_matrix_2, fill_value=0))
    return merge_matrix_1, target_v


def get_feature_matrix(data = 'hod', method = 'concatenate'):
    """

    :param data: the real part (default: hod)
    :param method: the imaginary part (default: concatenate)
    :return: dataframe, dataframe
    """
    if data == 'hod':
        PATH = './HumanObserved-Dataset/HumanObserved-Features-Data/'
    elif data == 'gsc':
        PATH = './GSC-Dataset/GSC-Features-Data/'

    same_pair = pd.read_csv(PATH + "same_pairs.csv", nrows=1000)
    diff_pair = pd.read_csv(PATH + "diffn_pairs.csv", nrows=1000)
    feature_data = pd.read_csv(PATH + PATH.rsplit('/', 2)[1] + ".csv")
    appended_data = _append_same_diff(same_pair, diff_pair)

    if method == 'concatenate':
        feature_m, target_v = _concat_features(appended_data, feature_data)
    elif method == 'subtract':
        feature_m, target_v = _subtract_features(appended_data, feature_data)


    feature_m = feature_m.loc[:, (feature_m != 0).any(axis=0)]
    return feature_m.values, target_v.values
