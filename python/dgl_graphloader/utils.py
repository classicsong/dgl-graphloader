"""Dataset utilities."""
from __future__ import absolute_import

import os
import sys
import errno
from multiprocessing import Manager,Process
import pickle

import numpy as np
import scipy.sparse as sp

import spacy
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

################### Basic operations  ########################
def save_info(path, info):
    """ Save dataset related information into disk.
    Parameters
    ----------
    path : str
        File to save information.
    info : dict
        A python dict storing information to save on disk.
    """
    with open(path, "wb" ) as pf:
        pickle.dump(info, pf)


def load_info(path):
    """ Load dataset related information from disk.
    Parameters
    ----------
    path : str
        File to load information from.
    Returns
    -------
    info : dict
        A python dict storing information loaded from disk.
    """
    with open(path, "rb") as pf:
        info = pickle.load(pf)
    return info

################### Feature Processing #######################

def field2idx(cols, fields):
    idx_cols = []
    # find index of each target field name
    for tg_field in cols:
        for i, field_name in enumerate(fields):
            if field_name == tg_field:
                idx_cols.append(i)
                break
    return idx_cols

def get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def row_normalize(features):
    mx = sp.csr_matrix(features, dtype=np.float32)

    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return np.array(mx.todense())

def col_normalize(features):
    mx = sp.csr_matrix(features, dtype=np.float32)

    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_inv[np.isnan(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv).transpose()
    mx = mx.dot(c_mat_inv)
    return np.array(mx.todense())

def float_row_l1_normalize(features):
    rowsum = np.sum(np.abs(features), axis=1)
    r_inv = np.power(rowsum, -1).reshape(-1,1)
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    return features * r_inv

def float_col_l1_normalize(features):
    colsum = np.sum(np.abs(features), axis=0)
    c_inv = np.power(colsum, -1)
    c_inv[np.isinf(c_inv)] = 0.
    c_inv[np.isnan(c_inv)] = 0.
    return features * c_inv

def float_col_maxmin_normalize(features):
    feats = np.transpose(features)
    min_val = np.reshape(np.amin(feats, axis=1), (-1, 1))
    max_val = np.reshape(np.amax(feats, axis=1), (-1, 1))
    norm = (feats - min_val) / (max_val - min_val)
    norm[np.isinf(norm)] = 0.
    norm[np.isnan(norm)] = 0.
    return np.transpose(norm)

def embed_word2vec(str_val, nlps):
    """ Use NLP encoder to encode the string into vector

    There can be multiple NLP encoders in nlps. Each encoder
    is invoded to generate a embedding for the input string and
    the resulting embeddings are concatenated.

    Parameters
    ----------
    str_val : str
        words to encode

    nlps : list of func
        a list of nlp encoder functions
    """
    vector = None
    for nlp in nlps:
        doc = nlp(str_val)
        if vector is None:
            vector = doc.vector
        else:
            vector = np.concatenate((vector, doc.vector))
    return vector

def parse_lang_feat(str_feats, nlp_encoders, verbose=False):
    """ Parse a list of strings using word2vec encoding using NLP encoders in nlps

    Parameters
    ----------
    str_feats : list of str
        list of strings to encode

    nlp_encoders : list of func
        a list of nlp encoder functions

    verbose : bool, optional
        print out debug info
        Default: False

    Return
    ------
    numpy.array
        the encoded features
    """
    features = []
    num_feats = len(str_feats)
    num_process = num_feats if num_feats < 8 else 8 # TODO(xiangsx) get system nproc
    batch_size = (num_feats + num_process - 1) // num_process

    def embed_lang(d, proc_idx, feats):
        res_feats = []
        for s_feat in feats:
            res_feats.append(embed_word2vec(s_feat, nlp_encoders))
        d[proc_idx] = res_feats

    # use multi process to process the feature
    manager = Manager()
    d = manager.dict()
    job=[]
    for i in range(num_process):
        sub_info = str_feats[i * batch_size : (i+1) * batch_size \
                         if (i+1) * batch_size < num_feats else num_feats]
        job.append(Process(target=embed_lang, args=(d, i, sub_info)))

    for p in job:
        p.start()

    for p in job:
        p.join()

    for i in range(num_process):
        if len(d[i]) > 0:
            features.append(d[i])

    features = np.concatenate(features)
    if verbose:
        print(features.shape)

    return features

def parse_word2vec_feature(str_feats, languages, verbose=False):
    """ Parse a list of strings using word2vec encoding using NLP encoders in nlps

    Parameters
    ----------
    str_feats : list of str
        list of strings to encode

    languages : list of string
        list of languages used to encode the feature string.

    verbose : bool, optional
        print out debug info
        Default: False

    Return
    ------
    numpy.array
        the encoded features

    Examples
    --------

    >>> inputs = ['hello', 'world']
    >>> languages = ['en_core_web_lg', 'fr_core_news_lg']
    >>> feats = parse_word2vec_node_feature(inputs, languages)

    """
    import spacy

    nlp_encoders = []
    for lang in languages:
        encoder = spacy.load(lang)
        nlp_encoders.append(encoder)

    return parse_lang_feat(str_feats, nlp_encoders, verbose)

def parse_category_single_feat(category_inputs, norm=None, classes=None):
    """ Parse categorical features and convert it into onehot encoding.

    Each entity of category_inputs should only contain only one category.

    Parameters
    ----------
    category_inputs : list of str
        input categorical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, sane as None
    classes : list
        predefined class list

    Note
    ----
    sklearn.preprocessing.LabelBinarizer is used to convert
    categorical features into a onehot encoding format.

    Return
    ------
    numpy.array
        The features in numpy array

    Examples
    --------

    >>> inputs = ['A', 'B', 'C', 'A']
    >>> feats = parse_category_single_feat(inputs)
    >>> feats
        array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

    """
    from sklearn.preprocessing import LabelBinarizer
    if classes is not None:
        lb = LabelBinarizer()
        lb.fit(classes)
        feat = lb.transform(category_inputs)
    else:
        lb = LabelBinarizer()
        feat = lb.fit_transform(category_inputs)

    # if there are only 2 catebories,
    # fit_transform only create a array of [0, 1, ...]
    if feat.shape[1] == 1:
        f = np.zeros((feat.shape[0], 2))
        f[range(f.shape[0]),feat.squeeze()] = 1.
        feat = f

    c_map = {i : c for i, c in enumerate(lb.classes_)}
    if norm == 'col':
        return col_normalize(feat), c_map
    else:
        return feat, c_map

def parse_category_multi_feat(category_inputs, norm=None, classes=None):
    """ Parse categorical features and convert it into multi-hot encoding.

    Each entity of category_inputs may contain multiple categorical labels.
    It uses multi-hot encoding to encode these labels.

    Parameters
    ----------
    category_inputs : list of list of str
        input categorical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, row-based normalization. Normalize the data for
        each row:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{x_{ij}}}

        Default: None
    classes : list
        predefined class list

    Note
    ----
    sklearn.preprocessing.MultiLabelBinarizer is used to convert
    categorical features into a multilabel format.

    Return
    ------
    numpy.array
        The features in numpy array

    Example
    -------

    >>> inputs = [['A', 'B', 'C',], ['A', 'B'], ['C'], ['A']]
    >>> feats = parse_category_multi_feat(inputs)
    >>> feats
        array([[1.,1.,1.],[1.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

    """
    from sklearn.preprocessing import MultiLabelBinarizer
    if classes is not None:
        mlb = MultiLabelBinarizer(classes=classes)
        feat = mlb.fit(category_inputs)
    else:
        mlb = MultiLabelBinarizer()
        feat = mlb.fit_transform(category_inputs)

    c_map = {i : c for i, c in enumerate(mlb.classes_)}
    if norm == 'col':
        return col_normalize(feat), c_map
    if norm == 'row':
        return row_normalize(feat), c_map
    else:
        return feat, c_map

def parse_numerical_feat(numerical_inputs, norm=None):
    """ Parse numerical features.

    Parameters
    ----------
    numerical_inputs : list of float or list of list of float
        input numerical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `standard`:, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{|x_{ij}|}}

        (3) `min-max`: column-based min-max normalization. Normalize the data
        for each column:

        .. math::
            norm_i = \frac{x_i - min(x[:])}{max(x[:])-min(x[:])}


    Return
    ------
    numpy.array
        The features in numpy array

    Example

    >>> inputs = [[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]]
    >>> feat = parse_numerical_feat(inputs, norm='col')
    >>> feat
    array([[0.25, 0., 0.],[0.5, 0.33333333, 0.25],[0.25, 0.66666667, -0.75]])

    """
    feat = np.array(numerical_inputs, dtype='float')

    if norm == 'standard':
        return float_col_l1_normalize(feat)
    elif norm == 'min-max':
        return float_col_maxmin_normalize(feat)
    else:
        return feat

def parse_numerical_multihot_feat(input_feats, low, high, bucket_cnt, window_size, norm=None):
    r""" Parse numerical features by matching them into
        different buckets.

    A bucket range based algorithm is used to convert numerical value into multi-hop
    encoding features.

    A numerical value range [low, high) is defined, and it is
    divied into #bucket_cnt buckets. For a input V, we get its effected range as
    [V - window_size/2, V + window_size/2] and check how many buckets it covers in
    [low, high).

    Parameters
    ----------
    input_feats : list of float
        Input numerical features
    low : float
        Lower bound of the range of the numerical values.
        All v_i < low will be set to v_i = low.
    high : float
        Upper bound of the range of the numerical values.
        All v_j > high will be set to v_j = high.
    bucket_cnt: int
        Number of bucket to use.
    slide_window_size: int
        The sliding window used to convert numerical value into bucket number.
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, row-based normalization. Normalize the data for
        each row:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{x_{ij}}}

    Example
    -------

    >>> inputs = [0., 15., 26., 40.]
    >>> low = 10.
    >>> high = 30.
    >>> bucket_cnt = 4
    >>> window_size = 10. # range is 10 ~ 15; 15 ~ 20; 20 ~ 25; 25 ~ 30
    >>> feat = parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    >>> feat
        array([[1., 0., 0., 0],
               [1., 1., 1., 0.],
               [0., 0., 1., 1.],
               [0., 0., 0., 1.]])

    """
    raw_feats = np.array(input_feats, dtype=np.float32)
    num_nodes = raw_feats.shape[0]
    feat = np.zeros((num_nodes, bucket_cnt), dtype=np.float32)

    bucket_size = (high - low) / bucket_cnt
    eposilon = bucket_size / 10
    low_val = raw_feats - window_size/2
    high_val = raw_feats + window_size/2
    low_val[low_val < low] = low
    high_val[high_val < low] = low
    high_val[high_val >= high] = high - eposilon
    low_val[low_val >= high] = high - eposilon
    low_val -= low
    high_val -= low
    low_idx = (low_val / bucket_size).astype('int')
    high_idx = (high_val / bucket_size).astype('int') + 1

    for i in range(raw_feats.shape[0]):
        idx = np.arange(start=low_idx[i], stop=high_idx[i])
        feat[i][idx] = 1.

    if norm == 'col':
        return col_normalize(feat)
    if norm == 'row':
        return row_normalize(feat)
    else:
        return feat
