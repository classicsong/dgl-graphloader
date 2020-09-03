import os
from pathlib import Path

import unittest, pytest
import numpy as np
import torch as th

import dgl_graphloader

def create_category_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}B{}A,B\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}{}A\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}B{}C,B\n".format(separator,separator,separator))
    node_feat_f.write("node3{}A{}C{}A,C\n".format(separator,separator,separator))
    node_feat_f.close()

def create_numerical_node_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3{}feat4\n".format(sep,sep,sep,sep))
    node_feat_f.write("node1{}1.{}2.{}0.{}1.,2.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node2{}2.{}-1.{}0.{}2.,-1.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}0.{}0.{}0.{}0.,0.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}4.{}-2.{}0.{}4.,-2.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.close()

def create_numerical_bucket_node_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}0.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}5.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}15.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}20.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}10.1\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}25.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}30.1\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}40.\n".format(sep,sep,sep))
    node_feat_f.close()

def create_numerical_edge_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_s{}node_d{}feat1\n".format(sep,sep))
    node_feat_f.write("node1{}node4{}1.\n".format(sep,sep))
    node_feat_f.write("node2{}node5{}2.\n".format(sep,sep))
    node_feat_f.write("node3{}node6{}0.\n".format(sep,sep))
    node_feat_f.write("node3{}node3{}4.\n".format(sep,sep))
    node_feat_f.close()

def create_word_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}B{}24\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}{}1\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}B{}12\n".format(separator,separator,separator))
    node_feat_f.write("node3{}A{}C{}13\n".format(separator,separator,separator))
    node_feat_f.close()

def create_multiple_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}0.1{}A,B\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}0.3{}A\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}0.2{}C,B\n".format(separator,separator,separator))
    node_feat_f.write("node4{}A{}-1.1{}A,C\n".format(separator,separator,separator))
    node_feat_f.close()

def create_multiple_edge_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_s{}node_d{}feat1{}feat2{}feat3\n".format(sep,sep,sep,sep))
    node_feat_f.write("node1{}node_a{}0.2{}0.1{}1.1\n".format(sep,sep,sep,sep))
    node_feat_f.write("node2{}node_b{}-0.3{}0.3{}1.2\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}node_c{}0.3{}0.2{}-1.2\n".format(sep,sep,sep,sep))
    node_feat_f.write("node4{}node_d{}-0.2{}-1.1{}0.9\n".format(sep,sep,sep,sep))
    node_feat_f.close()

def create_node_feats(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}label1{}label2\n".format(separator,separator))
    node_feat_f.write("node1{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node2{}A{}E,C,D\n".format(separator,separator))
    node_feat_f.write("node3{}C{}F,A,B\n".format(separator,separator))
    node_feat_f.write("node4{}A{}G,E\n".format(separator,separator))
    node_feat_f.write("node5{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node6{}C{}E,C,D\n".format(separator,separator))
    node_feat_f.write("node7{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node8{}A{}E,C,D\n".format(separator,separator))
    node_feat_f.close()

def create_node_labels(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}label1{}label2\n".format(separator,separator))
    node_feat_f.write("node1{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node2{}A{}E,C,D\n".format(separator,separator))
    node_feat_f.write("node3{}C{}F,A,B\n".format(separator,separator))
    node_feat_f.write("node4{}A{}G,E\n".format(separator,separator))
    node_feat_f.close()

def create_node_valid_labels(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}label1{}label2\n".format(separator,separator))
    node_feat_f.write("node5{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node6{}C{}E,C,D\n".format(separator,separator))
    node_feat_f.close()

def create_node_test_labels(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}label1{}label2\n".format(separator,separator))
    node_feat_f.write("node7{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node8{}A{}E,C,D\n".format(separator,separator))
    node_feat_f.close()

def create_edge_labels(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}label1{}label2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}node4{}A{}D,A\n".format(sep,sep,sep))
    node_feat_f.write("node2{}node3{}A{}E,C,D\n".format(sep,sep,sep))
    node_feat_f.write("node3{}node2{}C{}F,A,B\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node1{}A{}G,E\n".format(sep,sep,sep))
    node_feat_f.close()

def create_train_edge_labels(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}label1{}label2\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node2{}A{}D,A\n".format(sep,sep,sep))
    node_feat_f.write("node3{}node3{}A{}E,C,D\n".format(sep,sep,sep))
    node_feat_f.close()

def create_graph_edges(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}rel_1{}rel_2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}node2{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node2{}node1{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node3{}node1{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node3{}A{}B\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node4{}A{}A\n".format(sep,sep,sep))
    node_feat_f.close()

def create_graph_feat_edges(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}feat_1\n".format(sep,sep))
    node_feat_f.write("node1{}node4{}0.1\n".format(sep,sep))
    node_feat_f.write("node2{}node3{}0.2\n".format(sep,sep))
    node_feat_f.write("node3{}node2{}0.3\n".format(sep,sep))
    node_feat_f.write("node4{}node1{}0.4\n".format(sep,sep))
    node_feat_f.write("node1{}node2{}0.5\n".format(sep,sep))
    node_feat_f.write("node2{}node1{}0.6\n".format(sep,sep))
    node_feat_f.write("node3{}node1{}0.7\n".format(sep,sep))
    node_feat_f.write("node4{}node3{}0.8\n".format(sep,sep))
    node_feat_f.write("node4{}node4{}0.9\n".format(sep,sep))
    node_feat_f.close()

def create_multiple_label(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write(
        "node{}label1{}label2{}label3{}label4{}label5{}node_d{}node_d2{}node_d3\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node1{}A{}A{}C{}A,B{}A,C{}node3{}node1{}node4\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node2{}B{}B{}B{}A{}B{}node4{}node2{}node5\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node3{}C{}C{}A{}C,B{}A{}node5{}node1{}node6\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node4{}A{}A{}A{}A,C{}A,B{}node6{}node2{}node7\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.close()


def test_node_category_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_category_node_feat(Path(tmpdirname), 'node_category_feat.csv')

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1], feat_name='tf')
        feat_loader.addCategoryFeature(['node', 'feat1'], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1'], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,0],[1,0],[0,1],[1,0]]),
                           f_1[3])
        assert np.allclose(np.array([[1,0],[1,0],[0,1],[1,0]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,0],[1./3.,0],[0,1],[1./3.,0]]),
                           f_3[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1, 2])
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'nf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0,0.5,0.5],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1./2.,0],
                                     [1./3.,0,    0],
                                     [0,    1./2.,1./2.],
                                     [1./3.,0,    1./2.]]),
                           f_3[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1, 2], rows=[0,1,3])
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'],
                                        rows=[0,1,3], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'],
                                        rows=[0,1,3], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1.,0.],
                                     [1./3.,0.,0.],
                                     [1./3.,0.,1.]]),
                           f_3[3])


        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                                'node_category_feat.csv'))
        feat_loader.addMultiCategoryFeature([0, 3], separator=',')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',', norm='row', node_type='node')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',', norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0,0.5,0.5],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1./2.,0],
                                     [1./3.,0,    0],
                                     [0,    1./2.,1./2.],
                                     [1./3.,0,    1./2.]]),
                           f_3[3])

        feat_loader.addMultiCategoryFeature([0, 3], rows=[0,1,3], separator=',')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',',
                                            rows=[0,1,3], norm='row', node_type='node')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',',
                                            rows=[0,1,3], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1.,0.],
                                     [1./3.,0.,0.],
                                     [1./3.,0.,1.]]),
                           f_3[3])

def test_node_numerical_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_node_feat(Path(tmpdirname), 'node_numerical_feat.csv')

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0, 1])
        feat_loader.addNumericalFeature(['node', 'feat1'], norm='standard', node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1'], norm='min-max', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'nf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.],[2.],[0.],[4.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.],[2./7.],[0.],[4./7.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.],[2./4],[0.],[1.]]),
                           f_3[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2,3],feat_name='tf')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        norm='standard',
                                        node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        norm='min-max',
                                        node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.,2.,0.],[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.,2./5.,0.],[2./7.,-1./5.,0.],[0.,0.,0.],[4./7.,-2./5.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.,1.,0.],[2./4,1./4.,0.],[0.,2./4.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader.addNumericalFeature([0,1,2,3],rows=[1,2,3])
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        rows=[1,2,3],
                                        norm='standard',
                                        node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        rows=[1,2,3],
                                        norm='min-max',
                                        node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[2./6.,-1./3.,0.],[0.,0.,0.],[4./6.,-2./3.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[2./4.,1./2.,0.],[0.,1.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addMultiNumericalFeature([0,4], separator=',')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             norm='standard',
                                             node_type='node')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             norm='min-max',
                                             node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.,2.,0.],[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.,2./5.,0.],[2./7.,-1./5.,0.],[0.,0.,0.],[4./7.,-2./5.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.,1.,0.],[2./4,1./4.,0.],[0.,2./4.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader.addMultiNumericalFeature([0,4], separator=',', rows=[1,2,3])
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             rows=[1,2,3],
                                             norm='standard',
                                             node_type='node')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             rows=[1,2,3],
                                             norm='min-max',
                                             node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[2./6.,-1./3.,0.],[0.,0.,0.],[4./6.,-2./3.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[2./4.,1./2.,0.],[0.,1.,0.],[1.,0.,0.]]),
                           f_3[3])

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_bucket_node_feat(Path(tmpdirname), 'node_numerical_bucket_feat.csv')

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_bucket_feat.csv'))
        feat_loader.addNumericalBucketFeature([0, 2],
                                              feat_name='tf',
                                              range=[10,30],
                                              bucket_cnt=2)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.],
                                    [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.],
                                    [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4., 0.], [1./4., 0.], [1./4., 0.], [0., 1./4],
                                     [1./4., 0.], [0., 1./4.], [0., 1./4.], [0., 1./4.]]),
                           f_3[3])

        feat_loader.addNumericalBucketFeature([0, 2],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.],
                                     [1., 0.], [0., 1.], [0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.],
                                     [1., 0.], [0., 1.], [0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3., 0.], [1./3., 0.], [0., 1./3],
                                     [1./3., 0.], [0., 1./3.], [0., 1./3.]]),
                           f_3[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_bucket_feat.csv'))
        feat_loader.addNumericalBucketFeature([0, 2],
                                              feat_name='tf',
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0., 0., 0],
                                     [1., 0., 0., 0],
                                     [1., 1., 1., 0.],
                                     [0., 1., 1., 1.],
                                     [1., 1., 0., 0.],
                                     [0., 0., 1., 1.],
                                     [0., 0., 0., 1.],
                                     [0., 0., 0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0., 0., 0],
                                     [1., 0., 0., 0],
                                     [1./3., 1./3., 1./3., 0.],
                                     [0., 1./3., 1./3., 1./3.],
                                     [1./2., 1./2., 0., 0.],
                                     [0., 0., 1./2., 1./2.],
                                     [0., 0., 0., 1.],
                                     [0., 0., 0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4., 0.,    0.,    0],
                                     [1./4., 0.,    0.,    0],
                                     [1./4., 1./3., 1./3., 0.],
                                     [0.,    1./3., 1./3., 1./4.],
                                     [1./4., 1./3., 0.,    0.],
                                     [0.,    0.,    1./3., 1./4.],
                                     [0.,    0.,    0.,    1./4.],
                                     [0.,    0.,    0.,    1./4.]]),
                           f_3[3])

def test_edge_numerical_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_edge_feat(Path(tmpdirname), 'edge_numerical_feat.csv')

        feat_loader = dgl_graphloader.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0, 1, 2], feat_name='tf')
        feat_loader.addNumericalFeature(['node_s', 'node_d', 'feat1'],
                                        norm='standard',
                                        edge_type=('src', 'rel', 'dst'))
        feat_loader.addNumericalFeature(['node_d', 'node_s', 'feat1'],
                                        norm='min-max',
                                        edge_type=('dst', 'rev-rel', 'src'))
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'ef'
        assert f_3[0] == 'ef'
        assert f_1[1] is None
        assert f_2[1] == ('src', 'rel', 'dst')
        assert f_3[1] == ('dst', 'rev-rel', 'src')
        assert f_1[2] == f_2[2]
        assert f_1[2] == ['node1','node2','node3','node3']
        assert f_3[2] == ['node4','node5','node6','node3']
        assert f_1[3] == f_2[3]
        assert f_1[3] == ['node4','node5','node6','node3']
        assert f_3[3] == ['node1','node2','node3','node3']
        assert np.allclose(np.array([[1.],[2.],[0.],[4.]]),
                           f_1[4])
        assert np.allclose(np.array([[1./7.],[2./7.],[0.],[4./7.]]),
                           f_2[4])
        assert np.allclose(np.array([[1./4.],[2./4],[0.],[1.]]),
                           f_3[4])
        feat_loader.addNumericalFeature(['node_s', 'node_d', 'feat1'],
                                        rows=[1,2,3],
                                        norm='standard',
                                        edge_type=('src', 'rel', 'dst'))
        feat_loader.addNumericalFeature(['node_d', 'node_s', 'feat1'],
                                        rows=[1,2,3],
                                        norm='min-max',
                                        edge_type=('dst', 'rev-rel', 'src'))
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        assert f_1[1] == ('src', 'rel', 'dst')
        assert f_2[1] == ('dst', 'rev-rel', 'src')
        assert f_1[2] == ['node2','node3','node3']
        assert f_2[2] == ['node5','node6','node3']
        assert f_1[3] == ['node5','node6','node3']
        assert f_2[3] == ['node2','node3','node3']
        assert np.allclose(np.array([[2./6.],[0.],[4./6.]]),
                           f_1[4])
        assert np.allclose(np.array([[2./4],[0.],[1.]]),
                           f_2[4])

def test_node_word2vec_feature_loader():
    import tempfile
    import spacy
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_word_node_feat(Path(tmpdirname), 'node_word_feat.csv')

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_word_feat.csv'))
        feat_loader.addWord2VecFeature([0, 1], languages=['en_core_web_lg'], feat_name='tf')
        feat_loader.addWord2VecFeature(['node', 'feat1'],
                                       languages=['en_core_web_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat1'],
                                       languages=['en_core_web_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp = spacy.load('en_core_web_lg')
        assert np.allclose(np.array([nlp("A").vector,
                                     nlp("A").vector,
                                     nlp("C").vector,
                                     nlp("A").vector]),
                           f_1[3])

        feat_loader.addWord2VecFeature([0, 3], languages=['en_core_web_lg', 'fr_core_news_lg'])
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp1 = spacy.load('fr_core_news_lg')
        assert np.allclose(np.array([np.concatenate((nlp("24").vector, nlp1("24").vector)),
                                     np.concatenate((nlp("1").vector, nlp1("1").vector)),
                                     np.concatenate((nlp("12").vector, nlp1("12").vector)),
                                     np.concatenate((nlp("13").vector, nlp1("13").vector))]),
                           f_1[3])

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_word_feat.csv'))
        feat_loader.addWord2VecFeature([0, 3],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'])
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp1 = spacy.load('fr_core_news_lg')
        assert np.allclose(np.array([np.concatenate((nlp("1").vector, nlp1("1").vector)),
                                     np.concatenate((nlp("12").vector, nlp1("12").vector))]),
                           f_1[3])

def test_node_label_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_labels(Path(tmpdirname), 'labels.csv')
        label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'labels.csv'))
        label_loader.addTrainSet([0,1])
        label_loader.addValidSet(['node','label1'], node_type='node')
        label_loader.addTestSet(['node','label1'], rows=[0,2], node_type='node')
        label_loader.addSet(['node','label1'], [0.5, 0.25, 0.25], rows=[0,1,2,3], node_type='nt')
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == 'node'
        assert l_3[0] == 'node'
        assert l_4[0] == 'nt'
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == ['A','A','C','A']
        assert l_3[2] == ['A','C']
        assert l_4[2] == l_1[2]
        assert l_1[3] == (1., 0., 0.)
        assert l_2[3] == (0., 1., 0.)
        assert l_3[3] == (0., 0., 1.)
        assert l_4[3] == (0.5, 0.25, 0.25)

        label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'labels.csv'))
        label_loader.addTrainSet([0,2], multilabel=True, separator=',')
        label_loader.addValidSet(['node','label2'],
                                 multilabel=True,
                                 separator=',',
                                 node_type='node')
        label_loader.addTestSet(['node','label2'],
                                 multilabel=True,
                                 separator=',',
                                 rows=[0,2],
                                 node_type='node')
        label_loader.addSet(['node','label2'],
                            [0.5, 0.25, 0.25],
                            multilabel=True,
                            separator=',', rows=[0,1,2,3], node_type='nt')
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == 'node'
        assert l_3[0] == 'node'
        assert l_4[0] == 'nt'
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == [['D','A'],['E','C','D'],['F','A','B'],['G','E']]
        assert l_3[2] == [['D','A'],['F','A','B']]
        assert l_4[2] == l_1[2]
        assert l_1[3] == (1., 0., 0.)
        assert l_2[3] == (0., 1., 0.)
        assert l_3[3] == (0., 0., 1.)
        assert l_4[3] == (0.5, 0.25, 0.25)

def test_edge_label_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_edge_labels(Path(tmpdirname), 'edge_labels.csv')
        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_labels.csv'))
        label_loader.addTrainSet([0,1,2])
        label_loader.addValidSet(['node_0','node_1','label1'],
                                 edge_type=('src','rel','dst'))
        label_loader.addTestSet(['node_0','node_1','label1'],
                                rows=[0,2],
                                edge_type=('src','rel','dst'))
        label_loader.addSet(['node_0','node_1','label1'],
                            [0.5, 0.25, 0.25],
                            rows=[0,1,2,3],
                            edge_type=('src_n','rel_r','dst_n'))
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == ('src','rel','dst')
        assert l_3[0] == ('src','rel','dst')
        assert l_4[0] == ('src_n','rel_r','dst_n')
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == ['node4', 'node3', 'node2', 'node1']
        assert l_3[2] == ['node4', 'node2']
        assert l_4[2] == l_1[2]
        assert l_1[3] == l_2[3]
        assert l_1[3] == ['A','A','C','A']
        assert l_3[3] == ['A','C']
        assert l_4[3] == l_1[3]
        assert l_1[4] == (1., 0., 0.)
        assert l_2[4] == (0., 1., 0.)
        assert l_3[4] == (0., 0., 1.)
        assert l_4[4] == (0.5, 0.25, 0.25)

        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_labels.csv'))
        label_loader.addTrainSet([0,1,3], multilabel=True, separator=',')
        label_loader.addValidSet(['node_0','node_1','label2'],
                                 multilabel=True,
                                 separator=',',
                                 edge_type=('src','rel','dst'))
        label_loader.addTestSet(['node_0','node_1','label2'],
                                 multilabel=True,
                                 separator=',',
                                 rows=[0,2],
                                 edge_type=('src','rel','dst'))
        label_loader.addSet(['node_0','node_1','label2'],
                            [0.5, 0.25, 0.25],
                            multilabel=True,
                            separator=',',
                            rows=[0,1,2,3],
                            edge_type=('src_n','rel_r','dst_n'))
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == ('src','rel','dst')
        assert l_3[0] == ('src','rel','dst')
        assert l_4[0] == ('src_n','rel_r','dst_n')
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == ['node4', 'node3', 'node2', 'node1']
        assert l_3[2] == ['node4', 'node2']
        assert l_4[2] == l_1[2]
        assert l_1[3] == l_2[3]
        assert l_1[3] == [['D','A'],['E','C','D'],['F','A','B'],['G','E']]
        assert l_3[3] == [['D','A'],['F','A','B']]
        assert l_4[3] == l_1[3]
        assert l_1[4] == (1., 0., 0.)
        assert l_2[4] == (0., 1., 0.)
        assert l_3[4] == (0., 0., 1.)
        assert l_4[4] == (0.5, 0.25, 0.25)

def test_edge_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'graphs.csv')
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addEdges([0,1])
        edge_loader.addEdges(['node_0','node_1'])
        edge_loader.addEdges(['node_0','node_1'],
                             rows=np.array([1,2,3,4]),
                             edge_type=('src', 'edge', 'dst'))
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == None
        assert e_2[0] == None
        assert e_3[0] == ('src','edge','dst')
        assert e_1[1] == e_2[1]
        assert e_1[1] == ['node1', 'node2', 'node3', 'node4', 'node4']
        assert e_3[1] == ['node2', 'node3', 'node4', 'node4']
        assert e_1[2] == e_2[2]
        assert e_1[2] == ['node2', 'node1', 'node1', 'node3', 'node4']
        assert e_3[2] == ['node1', 'node1', 'node3', 'node4']

        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,2],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == ('src_t','A','dst_t')
        assert e_2[0] == ('src_t','A','dst_t')
        assert e_3[0] == ('src','A','dst')
        assert e_1[1] == e_2[1]
        assert e_1[1] == ['node1', 'node2', 'node3', 'node4', 'node4']
        assert e_3[1] == ['node2', 'node3', 'node4', 'node4']
        assert e_1[2] == e_2[2]
        assert e_1[2] == ['node2', 'node1', 'node1', 'node3', 'node4']
        assert e_3[2] == ['node1', 'node1', 'node3', 'node4']

        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,3],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == ('src_t','C','dst_t')
        assert e_2[0] == ('src_t','B','dst_t')
        assert e_3[0] == ('src_t','A','dst_t')
        e_4 = edge_loader._edges[3]
        e_5 = edge_loader._edges[4]
        e_6 = edge_loader._edges[5]
        assert e_4[0] == ('src_t','C','dst_t')
        assert e_5[0] == ('src_t','B','dst_t')
        assert e_6[0] == ('src_t','A','dst_t')
        assert e_1[1] == e_4[1]
        assert e_2[1] == e_5[1]
        assert e_3[1] == e_6[1]
        assert e_1[1] == ['node1', 'node2', 'node3']
        assert e_2[1] == ['node4']
        assert e_3[1] == ['node4']
        assert e_1[2] == e_4[2]
        assert e_2[2] == e_5[2]
        assert e_3[2] == e_6[2]
        assert e_1[2] == ['node2', 'node1', 'node1']
        assert e_2[2] == ['node3']
        assert e_3[2] == ['node4']
        e_7 = edge_loader._edges[6]
        e_8 = edge_loader._edges[7]
        e_9 = edge_loader._edges[8]
        assert e_7[0] == ('src','C','dst')
        assert e_8[0] == ('src','B','dst')
        assert e_9[0] == ('src','A','dst')
        assert e_7[1] == ['node2', 'node3']
        assert e_8[1] == ['node4']
        assert e_9[1] == ['node4']
        assert e_7[2] == ['node1', 'node1']
        assert e_8[2] == ['node3']
        assert e_9[2] == ['node4']

def test_node_feature_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_node_feat(Path(tmpdirname), 'node_feat.csv')

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_feat.csv'))
        feat_loader.addNumericalFeature([0,2],norm='standard')
        feat_loader.addCategoryFeature([0,1])
        feat_loader.addMultiCategoryFeature([0,3], separator=',')

        node_dicts = {}
        result = feat_loader.process(node_dicts)
        assert len(result) == 1
        nids, feats = result[None]['nf']
        assert np.allclose(np.array([0,1,2,3]), nids)
        assert np.allclose(np.concatenate([np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                                           np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)
        assert node_dicts[None]['node1'] == 0
        assert node_dicts[None]['node2'] == 1
        assert node_dicts[None]['node3'] == 2
        assert node_dicts[None]['node4'] == 3
        node_dicts = {None: {'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        nids, feats = result[None]['nf']
        assert np.allclose(np.array([3,2,1,0]), nids)
        assert np.allclose(np.concatenate([np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                                           np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)

        feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_feat.csv'))
        feat_loader.addCategoryFeature(['node','feat1'], node_type='n1')
        feat_loader.addMultiCategoryFeature(['node','feat3'], separator=',', node_type='n1')
        feat_loader.addNumericalFeature(['node','feat2'], norm='standard', node_type='n2')
        node_dicts = {'n2':{'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        assert len(result) == 2
        assert len(node_dicts) == 2
        nids, feats = result['n1']['nf']
        assert np.allclose(np.array([0,1,2,3]), nids)
        assert np.allclose(np.concatenate([np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)
        nids, feats = result['n2']['nf']
        assert np.allclose(np.array([3,2,1,0]), nids)
        assert np.allclose(np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                           feats)

def test_edge_feature_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_edge_feat(Path(tmpdirname), 'edge_feat.csv')

        feat_loader = dgl_graphloader.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2],norm='standard')
        feat_loader.addNumericalFeature([0,1,3],norm='min-max')
        feat_loader.addNumericalFeature([0,1,4])
        node_dicts = {}
        result = feat_loader.process(node_dicts)
        assert len(result) == 1
        snids, dnids, feats = result[None]['ef']
        assert np.allclose(np.array([0,1,2,3]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]]),
                                           np.array([[1.1],[1.2],[-1.2],[0.9]])],
                                           axis=1),
                           feats)
        assert node_dicts[None]['node1'] == 0
        assert node_dicts[None]['node2'] == 1
        assert node_dicts[None]['node3'] == 2
        assert node_dicts[None]['node4'] == 3
        node_dicts = {None: {'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        snids, dnids, feats = result[None]['ef']
        assert np.allclose(np.array([3,2,1,0]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]]),
                                           np.array([[1.1],[1.2],[-1.2],[0.9]])],
                                           axis=1),
                           feats)

        feat_loader = dgl_graphloader.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2],norm='standard',edge_type=('n0','r0','n1'))
        feat_loader.addNumericalFeature([0,1,3],norm='min-max',edge_type=('n0','r0','n1'))
        feat_loader.addNumericalFeature([0,1,4],edge_type=('n1','r1','n0'))
        node_dicts = {'n0':{'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        assert len(result) == 2
        snids, dnids, feats = result[('n0','r0','n1')]['ef']
        assert np.allclose(np.array([3,2,1,0]), snids)
        assert np.allclose(np.array([0,1,2,3]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]])],
                                           axis=1),
                           feats)
        snids, dnids, feats = result[('n1','r1','n0')]['ef']
        assert np.allclose(np.array([4,5,6,7]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.array([[1.1],[1.2],[-1.2],[0.9]]),
                           feats)

def test_node_label_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_label(Path(tmpdirname), 'node_label.csv')

        label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet([0,1])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_nids)
        assert valid_nids is None
        assert test_nids is None
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert valid_labels is None
        assert test_labels is None
        label_loader.addValidSet([0,2])
        label_loader.addTestSet([0,3])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_nids)
        assert np.array_equal(np.array([0,1,2,3]), valid_nids)
        assert np.array_equal(np.array([0,1,2,3]), test_nids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0],[1,0,0]]), test_labels)

        # test with node type
        label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet([0,1], node_type='n1')
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addValidSet([0,2], rows=[1,2,3], node_type='n1')
        label_loader.addTestSet([0,3], rows=[0,1,2], node_type='n1')
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert 'n1' in result
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result['n1']
        assert np.array_equal(np.array([3,2,1,0]), train_nids)
        assert np.array_equal(np.array([2,1,0]), valid_nids)
        assert np.array_equal(np.array([3,2,1]), test_nids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0]]), test_labels)

        # test multilabel
        # test with node type
        label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet(['node','label4'],
                                 multilabel=True,
                                 separator=',',
                                 node_type='n1')
        label_loader.addSet(['node', 'label5'],
                            split_rate=[0.,0.5,0.5],
                            multilabel=True,
                            separator=',',
                            node_type='n1')
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        np.random.seed(0)
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert 'n1' in result
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result['n1']
        label_map = label_loader.label_map
        rev_map = {val:key for key,val in label_map['n1'].items()}
        vl_truth = np.zeros((2,3),dtype='int32')
        vl_truth[0][rev_map['A']] = 1
        vl_truth[1][rev_map['A']] = 1
        vl_truth[1][rev_map['B']] = 1
        tl_truth = np.zeros((2,3),dtype='int32')
        tl_truth[0][rev_map['B']] = 1
        tl_truth[1][rev_map['A']] = 1
        tl_truth[1][rev_map['C']] = 1
        assert np.array_equal(np.array([3,2,1,0]), train_nids)
        assert np.array_equal(np.array([1,0]), valid_nids)
        assert np.array_equal(np.array([2,3]), test_nids)
        assert np.array_equal(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]), train_labels)
        assert np.array_equal(vl_truth, valid_labels)
        assert np.array_equal(tl_truth, test_labels)

def test_edge_label_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_label(Path(tmpdirname), 'edge_label.csv')

        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        # only existence of the edge
        label_loader.addTrainSet([0,6])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_snids)
        assert np.array_equal(np.array([2,3,4,5]), train_dnids)
        assert valid_snids is None
        assert valid_dnids is None
        assert test_snids is None
        assert test_dnids is None
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        label_loader.addValidSet([0,7])
        label_loader.addTestSet([6,8])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_snids)
        assert np.array_equal(np.array([2,3,4,5]), train_dnids)
        assert np.array_equal(np.array([0,1,2,3]), valid_snids)
        assert np.array_equal(np.array([0,1,0,1]), valid_dnids)
        assert np.array_equal(np.array([2,3,4,5]), test_snids)
        assert np.array_equal(np.array([3,4,5,6]), test_dnids)

        # with labels
        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        label_loader.addTrainSet([0,6,1], edge_type=('n1', 'like', 'n1'))
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addValidSet(['node', 'node_d2', 'label2'], rows=[1,2,3], edge_type=('n1', 'like', 'n1'))
        label_loader.addTestSet(['node_d', 'node_d3', 'label3'], rows=[0,1,2], edge_type=('n1', 'like', 'n1'))
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert ('n1', 'like', 'n1') in result
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('n1', 'like', 'n1')]
        assert np.array_equal(np.array([3,2,1,0]), train_snids)
        assert np.array_equal(np.array([1,0,4,5]), train_dnids)
        assert np.array_equal(np.array([2,1,0]), valid_snids)
        assert np.array_equal(np.array([2,3,2]), valid_dnids)
        assert np.array_equal(np.array([1,0,4]), test_snids)
        assert np.array_equal(np.array([0,4,5]), test_dnids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0]]), test_labels)

        # with multiple labels
        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        label_loader.addTrainSet(['node','node_d','label4'],
                                 multilabel=True,
                                 separator=',',
                                 edge_type=('n1', 'like', 'n2'))
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addSet(['node_d2', 'node_d3', 'label5'],
                            split_rate=[0.,0.5,0.5],
                            multilabel=True,
                            separator=',',
                            edge_type=('n1', 'like', 'n2'))
        np.random.seed(0)
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert ('n1', 'like', 'n2') in result
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('n1', 'like', 'n2')]
        label_map = label_loader.label_map
        rev_map = {val:key for key,val in label_map[('n1', 'like', 'n2')].items()}
        vl_truth = np.zeros((2,3),dtype='int32')
        vl_truth[0][rev_map['A']] = 1
        vl_truth[1][rev_map['A']] = 1
        vl_truth[1][rev_map['B']] = 1
        tl_truth = np.zeros((2,3),dtype='int32')
        tl_truth[0][rev_map['B']] = 1
        tl_truth[1][rev_map['A']] = 1
        tl_truth[1][rev_map['C']] = 1
        assert np.array_equal(np.array([3,2,1,0]), train_snids)
        assert np.array_equal(np.array([0,1,2,3]), train_dnids)
        assert np.array_equal(np.array([3,2]), valid_snids)
        assert np.array_equal(np.array([3,4]), valid_dnids)
        assert np.array_equal(np.array([2,3]), test_snids)
        assert np.array_equal(np.array([2,1]), test_dnids)
        assert np.array_equal(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]), train_labels)
        assert np.array_equal(vl_truth, valid_labels)
        assert np.array_equal(tl_truth, test_labels)

def test_relation_edge_label_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'edge_label.csv')

        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        # only existence of the edge
        label_loader.addRelationalTrainSet([0,1,2])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','A','node')]
        assert np.array_equal(np.array([0,1,2,3,3]), train_snids)
        assert np.array_equal(np.array([1,0,0,2,3]), train_dnids)
        assert valid_snids is None
        assert valid_dnids is None
        assert test_snids is None
        assert test_dnids is None
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        label_loader.addRelationalValidSet([0,1,3],rows=[0,3])
        label_loader.addRelationalTestSet([0,1,3],rows=[1,2,4])
        result = label_loader.process(node_dicts)
        assert len(result) == 3
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','A','node')]
        assert np.array_equal(np.array([0,1,2,3,3]), train_snids)
        assert np.array_equal(np.array([1,0,0,2,3]), train_dnids)
        assert valid_snids is None
        assert valid_dnids is None
        assert np.array_equal(np.array([3]), test_snids)
        assert np.array_equal(np.array([3]), test_dnids)
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','C','node')]
        assert train_snids is None
        assert train_dnids is None
        assert np.array_equal(np.array([0]), valid_snids)
        assert np.array_equal(np.array([1]), valid_dnids)
        assert np.array_equal(np.array([1,2]), test_snids)
        assert np.array_equal(np.array([0,0]), test_dnids)
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','B','node')]
        assert train_snids is None
        assert train_dnids is None
        assert np.array_equal(np.array([3]), valid_snids)
        assert np.array_equal(np.array([2]), valid_dnids)
        assert test_snids is None
        assert test_dnids is None
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None

        np.random.seed(0)
        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        label_loader.addRelationalTrainSet([0,1,2])
        label_loader.addRelationalSet([0,1,3], split_rate=[0.,0.4,0.6])
        result = label_loader.process(node_dicts)
        assert len(result) == 3
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','A','node')]
        assert np.array_equal(np.array([0,1,2,3,3]), train_snids)
        assert np.array_equal(np.array([1,0,0,2,3]), train_dnids)
        assert np.array_equal(np.array([3]), test_snids)
        assert np.array_equal(np.array([3]), test_dnids)
        assert valid_snids is None
        assert valid_dnids is None
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','C','node')]
        assert train_snids is None
        assert train_dnids is None
        assert np.array_equal(np.array([2]), valid_snids)
        assert np.array_equal(np.array([0]), valid_dnids)
        assert np.array_equal(np.array([1,0]), test_snids)
        assert np.array_equal(np.array([0,1]), test_dnids)
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('node','B','node')]
        assert train_snids is None
        assert train_dnids is None
        assert valid_snids is None
        assert valid_dnids is None
        assert np.array_equal(np.array([3]), test_snids)
        assert np.array_equal(np.array([2]), test_dnids)
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None

def test_edge_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'graphs.csv')

        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))

        edge_loader.addEdges([0,1])
        edge_loader.addEdges(['node_0','node_1'])
        edge_loader.addEdges(['node_0','node_1'],
                             rows=np.array([1,2,3,4]),
                             edge_type=('src', 'edge', 'src'))
        node_dicts = {}
        result = edge_loader.process(node_dicts)
        assert len(result) == 2
        snids, dnids = result[None]
        assert np.array_equal(np.array([0,1,2,3,3,0,1,2,3,3]), snids)
        assert np.array_equal(np.array([1,0,0,2,3,1,0,0,2,3]), dnids)
        snids, dnids = result[('src', 'edge', 'src')]
        assert np.array_equal(np.array([0,1,2,2]), snids)
        assert np.array_equal(np.array([3,3,1,2]), dnids)

        # with categorical relation
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,2],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        node_dicts = {'src_t':{'node1':3,
                                'node2':2,
                                'node3':1,
                                'node4':0}}
        result = edge_loader.process(node_dicts)
        assert len(result) == 4
        snids, dnids = result[('src_t','A','dst_t')]
        assert np.array_equal(np.array([3,2,1,0,0,0]), snids)
        assert np.array_equal(np.array([0,1,1,2,3,3]), dnids)
        snids, dnids = result[('src_t','B','dst_t')]
        assert np.array_equal(np.array([0]), snids)
        assert np.array_equal(np.array([2]), dnids)
        snids, dnids = result[('src_t','C','dst_t')]
        assert np.array_equal(np.array([3,2,1]), snids)
        assert np.array_equal(np.array([0,1,1]), dnids)
        snids, dnids = result[('src','A','dst')]
        assert np.array_equal(np.array([0,1,2,2]), snids)
        assert np.array_equal(np.array([0,0,1,2]), dnids)

def test_build_graph():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'edges.csv')
        create_edge_labels(Path(tmpdirname), 'edge_labels.csv')
        create_node_labels(Path(tmpdirname), 'node_labels.csv')

        # homogeneous graph loader (edge labels)
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1])
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1,2],split_rate=[0.5,0.25,0.25])
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1])

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.process()

        node_id_map = graphloader.node2id
        assert None in node_id_map
        assert len(node_id_map[None]) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert node_id_map[None][key] == idx
        id_node_map = graphloader.id2node
        assert None in id_node_map
        assert len(id_node_map[None]) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert id_node_map[None][idx] == key
        label_map = graphloader.label_map
        assert len(label_map[None]) == 2
        assert label_map[None][0] == 'A'
        assert label_map[None][1] == 'C'

        g = graphloader.graph
        assert g.num_edges() == 9
        assert np.array_equal(g.edata['labels'].long().numpy(),
            np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[0,1],[1,0],[1,0],[1,0]]))
        assert th.nonzero(g.edata['train_mask']).shape[0] == 2
        assert th.nonzero(g.edata['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edata['test_mask']).shape[0] == 1
        assert np.allclose(g.ndata['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))

        # heterogeneous graph loader (edge labels)
        create_train_edge_labels(Path(tmpdirname), 'edge_train_labels.csv')
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1,2],split_rate=[0.5,0.25,0.25], edge_type=('a', 'follow', 'b'))
        edge_train_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_train_labels.csv'))
        edge_train_label_loader.addTrainSet([0,1,2], edge_type=('a', 'follow', 'b'))
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1], edge_type=('a', 'follow', 'b'))
        node_feat_loader2 = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader2.addCategoryFeature([0,1], node_type='b')
        edge_loader2 = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader2.addEdges([0,1], edge_type=('b', 'follow', 'a'))

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendEdge(edge_loader2)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendLabel(edge_train_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.appendFeature(node_feat_loader2)
        graphloader.process()

        node_id_map = graphloader.node2id
        assert 'a' in node_id_map
        assert len(node_id_map['a']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert node_id_map['a'][key] == idx
        id_node_map = graphloader.id2node
        assert 'a' in id_node_map
        assert len(id_node_map['a']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert id_node_map['a'][idx] == key
        assert 'b' in node_id_map
        assert len(node_id_map['b']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert node_id_map['b'][key] == idx
        assert 'b' in id_node_map
        assert len(id_node_map['b']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert id_node_map['b'][idx] == key

        label_map = graphloader.label_map
        assert len(label_map[('a', 'follow', 'b')]) == 2
        assert label_map[('a', 'follow', 'b')][0] == 'A'
        assert label_map[('a', 'follow', 'b')][1] == 'C'

        g = graphloader.graph
        assert g.num_edges(('a', 'follow', 'b')) == 11
        assert g.num_edges(('b', 'follow', 'a')) == 5
        assert np.array_equal(g.edges[('a', 'follow', 'b')].data['labels'].long().numpy(),
            np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0]]))
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['train_mask']).shape[0] == 4
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['test_mask']).shape[0] == 1
        assert np.allclose(g.nodes['a'].data['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))
        assert np.allclose(g.nodes['b'].data['nf'].numpy(),
            np.array([[1.,0.,],[1.,0.],[0.,1.],[1.,0.]]))

        # edge feat with edge labels
        create_graph_feat_edges(Path(tmpdirname), 'edges_feats.csv')
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1,2],split_rate=[0.5,0.25,0.25], edge_type=('a', 'follow', 'b'))
        edge_feat_loader = dgl_graphloader.EdgeFeatureLoader(os.path.join(tmpdirname, 'edges_feats.csv'))
        edge_feat_loader.addNumericalFeature([0,1,2], edge_type=('a', 'follow', 'b'))
        node_feat_loader2 = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader2.addCategoryFeature([0,1], node_type='b')
        edge_loader2 = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader2.addEdges([0,1], edge_type=('b', 'follow', 'a'))

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader2)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendFeature(edge_feat_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.appendFeature(node_feat_loader2)
        graphloader.process()
        node_id_map = graphloader.node2id
        assert 'b' in node_id_map
        assert len(node_id_map['b']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert node_id_map['b'][key] == idx
        id_node_map = graphloader.id2node
        assert 'b' in id_node_map
        assert len(id_node_map['b']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert id_node_map['b'][idx] == key
        assert 'a' in node_id_map
        assert len(node_id_map['a']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert node_id_map['a'][key] == idx
        assert 'a' in id_node_map
        assert len(id_node_map['a']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert id_node_map['a'][idx] == key

        g = graphloader.graph
        assert g.num_edges(('a', 'follow', 'b')) == 9
        assert g.num_edges(('b', 'follow', 'a')) == 5
        assert np.array_equal(g.edges[('a', 'follow', 'b')].data['labels'].long().numpy(),
            np.array([[1,0],[1,0],[0,1],[1,0],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]))
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['train_mask']).shape[0] == 2
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['test_mask']).shape[0] == 1
        assert np.array_equal(g.edges[('a', 'follow', 'b')].data['ef'].numpy(),
            np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]]))

        # heterogeneous graph loader (edge no labels)
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1],split_rate=[0.5,0.25,0.25], edge_type=('a', 'follow', 'b'))
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1], edge_type=('a', 'follow', 'b'))
        node_feat_loader2 = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader2.addCategoryFeature([0,1], node_type='b')
        edge_loader2 = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader2.addEdges([0,1], edge_type=('b', 'follow', 'a'))

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendEdge(edge_loader2)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.appendFeature(node_feat_loader2)
        graphloader.process()

        label_map = graphloader.label_map
        assert len(label_map) == 0
        g = graphloader.graph
        assert g.num_edges(('a', 'follow', 'b')) == 9
        assert g.num_edges(('b', 'follow', 'a')) == 5
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['train_mask']).shape[0] == 2
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['test_mask']).shape[0] == 1
        assert np.allclose(g.nodes['a'].data['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))
        assert np.allclose(g.nodes['b'].data['nf'].numpy(),
            np.array([[1.,0.,],[1.,0.],[0.,1.],[1.,0.]]))

        # heterogeneous graph loader (node labels)
        create_node_valid_labels(Path(tmpdirname), 'node_valid.csv')
        create_node_test_labels(Path(tmpdirname), 'node_test.csv')
        create_node_feats(Path(tmpdirname), 'node_feat.csv')
        node_label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_label_loader.addTrainSet([0,1], node_type='a')
        valid_label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname, 'node_valid.csv'))
        valid_label_loader.addValidSet([0,1], node_type='a')
        test_label_loader = dgl_graphloader.NodeLabelLoader(os.path.join(tmpdirname, 'node_test.csv'))
        test_label_loader.addTestSet([0,1], node_type='a')
        edge_feat_loader = dgl_graphloader.EdgeFeatureLoader(os.path.join(tmpdirname, 'edges_feats.csv'))
        edge_feat_loader.addNumericalFeature([0,1,2], edge_type=('a', 'in', 'aa'))
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1], edge_type=('a', 'follow', 'a'))
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_feat.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')


        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendLabel(node_label_loader)
        graphloader.appendLabel(valid_label_loader)
        graphloader.appendLabel(test_label_loader)
        graphloader.appendFeature(edge_feat_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.process()

        node_id_map = graphloader.node2id
        assert 'a' in node_id_map
        assert len(node_id_map['a']) == 8
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']):
            assert node_id_map['a'][key] == idx
        id_node_map = graphloader.id2node
        assert 'a' in id_node_map
        assert len(id_node_map['a']) == 8
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']):
            assert id_node_map['a'][idx] == key
        assert 'aa' in node_id_map
        assert len(node_id_map['aa']) == 4
        for idx, key in enumerate(['node4', 'node3', 'node2', 'node1']):
            assert node_id_map['aa'][key] == idx
        assert 'aa' in id_node_map
        assert len(id_node_map['aa']) == 4
        for idx, key in enumerate(['node4', 'node3', 'node2', 'node1']):
            assert id_node_map['aa'][idx] == key

        label_map = graphloader.label_map
        assert len(label_map['a']) == 2
        assert label_map['a'][0] == 'A'
        assert label_map['a'][1] == 'C'
        g = graphloader.graph
        assert g.num_edges(('a', 'in', 'aa')) == 9
        assert g.num_edges(('a', 'follow', 'a')) == 5
        assert np.array_equal(g.nodes['a'].data['train_mask'].long().numpy(), np.array([1,1,1,1,0,0,0,0]))
        assert np.array_equal(g.nodes['a'].data['valid_mask'].long().numpy(), np.array([0,0,0,0,1,1,0,0]))
        assert np.array_equal(g.nodes['a'].data['test_mask'].long().numpy(), np.array([0,0,0,0,0,0,1,1]))
        assert np.allclose(g.nodes['a'].data['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],
                      [0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1],
                      [1,0,1,0,0,1,0,0,0],[0,1,0,0,1,1,1,0,0],
                      [1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0]]))
        assert np.array_equal(g.edges[('a', 'in', 'aa')].data['ef'].numpy(),
            np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]]))

def test_add_reverse_edge():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'edges.csv')
        create_edge_labels(Path(tmpdirname), 'edge_labels.csv')
        create_node_labels(Path(tmpdirname), 'node_labels.csv')
        create_train_edge_labels(Path(tmpdirname), 'edge_train_labels.csv')

        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1,2],split_rate=[0.5,0.25,0.25], edge_type=('a', 'follow', 'b'))
        edge_train_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_train_labels.csv'))
        edge_train_label_loader.addTrainSet([0,1,2], edge_type=('a', 'follow', 'b'))
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1], edge_type=('a', 'follow', 'b'))
        node_feat_loader2 = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader2.addCategoryFeature([0,1], node_type='b')
        edge_loader2 = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader2.addEdges([0,1], edge_type=('b', 'follow', 'a'))

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendEdge(edge_loader2)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendLabel(edge_train_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.appendFeature(node_feat_loader2)
        graphloader.addReverseEdge()
        graphloader.process()

        node_id_map = graphloader.node2id
        assert 'a' in node_id_map
        assert len(node_id_map['a']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert node_id_map['a'][key] == idx
        id_node_map = graphloader.id2node
        assert 'a' in id_node_map
        assert len(id_node_map['a']) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert id_node_map['a'][idx] == key
        assert 'b' in node_id_map
        assert len(node_id_map['b']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert node_id_map['b'][key] == idx
        assert 'b' in id_node_map
        assert len(id_node_map['b']) == 4
        for idx, key in enumerate(['node2', 'node1', 'node3', 'node4']):
            assert id_node_map['b'][idx] == key

        label_map = graphloader.label_map
        assert len(label_map[('a', 'follow', 'b')]) == 2
        assert label_map[('a', 'follow', 'b')][0] == 'A'
        assert label_map[('a', 'follow', 'b')][1] == 'C'
        g = graphloader.graph
        assert g.num_edges(('a', 'follow', 'b')) == 11
        assert g.num_edges(('b', 'follow', 'a')) == 5
        assert g.num_edges(('b', 'rev-follow', 'a')) == 11
        assert g.num_edges(('a', 'rev-follow', 'b')) == 5
        assert 'labels' in g.edges[('a', 'follow', 'b')].data
        assert 'labels' not in g.edges[('b', 'rev-follow', 'a')].data
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['train_mask']).shape[0] == 4
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['test_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_train_mask']).shape[0] == 4
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_test_mask']).shape[0] == 1
        assert np.allclose(g.nodes['a'].data['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))
        assert np.allclose(g.nodes['b'].data['nf'].numpy(),
            np.array([[1.,0.,],[1.,0.],[0.,1.],[1.,0.]]))

        # heterogeneous graph loader (edge no labels)
        node_feat_loader = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1], node_type='a')
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',', node_type='a')
        edge_label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1],split_rate=[0.5,0.25,0.25], edge_type=('a', 'follow', 'b'))
        edge_loader = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1], edge_type=('a', 'follow', 'b'))
        node_feat_loader2 = dgl_graphloader.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader2.addCategoryFeature([0,1], node_type='b')
        edge_loader2 = dgl_graphloader.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader2.addEdges([0,1], edge_type=('b', 'follow', 'a'))

        np.random.seed(0)
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendEdge(edge_loader2)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.appendFeature(node_feat_loader2)
        graphloader.addReverseEdge()
        graphloader.process()

        label_map = graphloader.label_map
        assert len(label_map) == 0
        g = graphloader.graph
        assert g.num_edges(('a', 'follow', 'b')) == 9
        assert g.num_edges(('b', 'follow', 'a')) == 5
        assert g.num_edges(('b', 'rev-follow', 'a')) == 9
        assert g.num_edges(('a', 'rev-follow', 'b')) == 5
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['train_mask']).shape[0] == 2
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('a', 'follow', 'b')].data['test_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_train_mask']).shape[0] == 2
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('b', 'rev-follow', 'a')].data['rev_test_mask']).shape[0] == 1
        assert np.allclose(g.nodes['a'].data['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))
        assert np.allclose(g.nodes['b'].data['nf'].numpy(),
            np.array([[1.,0.,],[1.,0.],[0.,1.],[1.,0.]]))

        create_graph_edges(Path(tmpdirname), 'edge_label.csv')
        label_loader = dgl_graphloader.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        # only existence of the edge
        label_loader.addRelationalTrainSet([0,1,2],rows=[0,1,2,3])
        label_loader.addRelationalTrainSet([0,1,3],rows=[2,3])
        label_loader.addRelationalValidSet([0,1,3],rows=[0])
        label_loader.addRelationalTestSet([0,1,3],rows=[1,4])
        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendLabel(label_loader)
        graphloader.addReverseEdge()
        graphloader.process()
        label_map = graphloader.label_map
        assert len(label_map) == 0
        g = graphloader.graph
        assert th.nonzero(g.edges[('node','A','node')].data['train_mask']).shape[0] == 4
        assert th.nonzero(g.edges[('node','A','node')].data['valid_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','A','node')].data['test_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','rev-A','node')].data['rev_train_mask']).shape[0] == 4
        assert th.nonzero(g.edges[('node','rev-A','node')].data['rev_valid_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','rev-A','node')].data['rev_test_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','B','node')].data['train_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','B','node')].data['valid_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','B','node')].data['test_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','rev-B','node')].data['rev_train_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','rev-B','node')].data['rev_valid_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','rev-B','node')].data['rev_test_mask']).shape[0] == 0
        assert th.nonzero(g.edges[('node','C','node')].data['train_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','C','node')].data['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','C','node')].data['test_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','rev-C','node')].data['rev_train_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','rev-C','node')].data['rev_valid_mask']).shape[0] == 1
        assert th.nonzero(g.edges[('node','rev-C','node')].data['rev_test_mask']).shape[0] == 1

if __name__ == '__main__':
    # test Feature Loader
    test_node_category_feature_loader()
    test_node_numerical_feature_loader()
    #test_node_word2vec_feature_loader()
    test_edge_numerical_feature_loader()
    # test Label Loader
    test_node_label_loader()
    test_edge_label_loader()
    # test Edge Loader
    test_edge_loader()

    # test feature process
    test_node_feature_process()
    test_edge_feature_process()
    # test label process
    test_node_label_process()
    test_edge_label_process()
    test_relation_edge_label_process()
    # test edge process
    test_edge_process()

    test_build_graph()
    test_add_reverse_edge()
