""" Classes for loading raw graph"""
import os
import csv

import numpy as np
# TODO(xiangsx): Framework agnostic later
import torch as th
import dgl

from dgl.data import save_graphs, load_graphs
from .utils import save_info, load_info
from .utils import field2idx, get_id
from .csv_feature_loader import NodeFeatureLoader, EdgeFeatureLoader
from .csv_feature_loader import EdgeLoader
from .csv_label_loader import NodeLabelLoader, EdgeLabelLoader

class GraphLoader(object):
    r""" Generate DGLGraph by parsing files.

    GraphLoader generate DGLGraph by collecting EdgeLoaders,
    FeatureLoaders and LabelLoders and parse them iteratively.

    Parameters
    ----------
    name: str, optional
        name of the graph.
        default: 'graph'
    edge_loader: list of EdgeLoader
        edge loaders to load graph edges
        default: None
    feature_loader: list of NodeFeatureLoader and EdgeFeatureLoader
        feature loaders to load graph features and edges
        default: None
    label_loader: list of NodeLabelLoader and EdgeLabelLoader
        label loaders to load labels/ground-truth and edges
        default: None
    verbose: bool, optional
        Whether print debug info during parsing
        Default: False

    Note:
    -----

    **EdgeLoader is used to add edges that neither have features nor appeared in
    edge labels.** If one edge appears in both appendEdge and appendLabel, it will
    be added twice. **But if one edge appears in both EdgeFeatureLoader and
    EdgeLabelLoader, it will only be added once.


    Example:

    ** Create a Graph Loader **

    >>> graphloader = dgl.data.GraphLoader(name='example')

    """
    def __init__(self, name='graph',
        edge_loader=None, feature_loader=None, label_loader=None, verbose=False):
        self._name = name

        if edge_loader is not None:
            if not isinstance(edge_loader, list):
                raise RuntimeError("edge loaders should be a list of EdgeLoader")

            self._edge_loader = edge_loader
        else:
            self._edge_loader = []

        if feature_loader is not None:
            if not isinstance(feature_loader, list):
                raise RuntimeError("feature loaders should be " \
                    "a list of NodeFeatureLoader and EdgeFeatureLoader")

            self._feature_loader = feature_loader
        else:
            self._feature_loader = []

        if label_loader is not None:
            if not isinstance(label_loader, list):
                raise RuntimeError("label loaders should be " \
                    "a list of NodeLabelLoader and EdgeLabelLoader")

            self._label_loader = label_loader
        else:
            self._label_loader = []

        self._graph = None
        self._verbose = verbose
        self._node_dict = {}
        self._label_map = None

    def appendEdge(self, edge_loader):
        """ Add edges into graph

        Parameters
        ----------
        edge_loader: EdgeLoader
            edge loaders to load graph edges
            default: None

        Example:

        ** create edge loader to load edges **

        >>> edge_loader = dgl.data.EdgeLoader(input='edge.csv',
                                            separator="|")
        >>> edge_loader.addEdges([0, 1])

        ** Append edges into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendEdge(edge_loader)

        """
        if not isinstance(edge_loader, EdgeLoader):
            raise RuntimeError("edge loader should be a EdgeLoader")
        self._edge_loader.append(edge_loader)

    def appendFeature(self, feature_loader):
        """ Add features and edges into graph

        Parameters
        ----------
        feature_loader: NodeFeatureLoader or EdgeFeatureLoader
            feature loaders to load graph edges
            default: None

        Example:

        ** Creat a FeatureLoader to load user features from u.csv.**

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|")
        >>> user_loader.addCategoryFeature(cols=["id", "gender"], node_type='user')

        ** Append features into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendFeature(user_loader)

        """
        if not isinstance(feature_loader, NodeFeatureLoader) and \
            not isinstance(feature_loader, EdgeFeatureLoader):
            raise RuntimeError("feature loader should be a NodeFeatureLoader or EdgeFeatureLoader.")
        self._feature_loader.append(feature_loader)

    def appendLabel(self, label_loader):
        """ Add labels and edges into graph

        Parameters
        ----------
        label_loader: NodeLabelLoader or EdgeLabelLoader
            label loaders to load graph edges
            default: None

        Note
        ----
        To keep the overall design of the GraphLoader simple,
        it accepts only one LabelLoader.

        Examples:

        ** create node label loader to load labels **

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="|")
        >>> label_loader.addTrainSet([0, 1], rows=np.arange(start=0,
                                                            stop=100))

        ** Append labels into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendLabel(label_loader)

        """
        if not isinstance(label_loader, NodeLabelLoader) and \
            not isinstance(label_loader, EdgeLabelLoader):
            raise RuntimeError("label loader should be a NodeLabelLoader or EdgeLabelLoader.")
        self._label_loader.append(label_loader)

    def addReverseEdge(self):
        """ Add Reverse edges with new relation type.

        addReverseEdge works for heterogenous graphs. It adds
        a new relation type for each existing relation. For
        example, with relation ('head', 'rel', 'tail'), it will
        create a new relation type ('tail', 'rev-rel', 'head')
        and adds edges belong to ('head', 'rel', 'tail') into
        new relation type with reversed head and tail entity order.

        Example:

        ** create edge loader to load edges **

        >>> edge_loader = dgl.data.EdgeLoader(input='edge.csv',
                                            separator="|")
        >>> edge_loader.addEdges([0, 1],
                                src_type='user',
                                edge_type='likes',
                                dst_type='movie')

        ** Append edges into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendEdge(edge_loader)

        ** add reversed edges into graph **

        >>> graphloader.addReverseEdge()

        """
        assert self._g.is_homogeneous is False, \
            'Add reversed edges only work for heterogeneous graph'

        new_g = dgl.add_reverse_edges(self._g, copy_ndata=True, copy_edata=False)
        for etype in self._g.canonical_etypes:
            new_g.edges[etype].data = self._g.edges[etype].data
        self._g = new_g

    def process(self):
        """ Parsing EdgeLoaders, FeatureLoaders and LabelLoders to build the DGLGraph
        """
        graphs = {} # edge_type: (s, d, feat)
        nodes = {}
        edge_feat_results = []
        node_feat_results = []
        edge_label_results = []
        node_label_results = []

        if self._verbose:
            print('Start processing graph structure ...')
        # we first handle edges
        for edge_loader in self._edge_loader:
            # {edge_type: (snids, dnids)}
            edge_result = edge_loader.process(self._node_dict)
            for edge_type, vals in edge_result.items():
                snids, dnids = vals
                if edge_type in graphs:
                    graphs[edge_type] = (np.concatenate((graphs[edge_type][0], snids)),
                                         np.concatenate((graphs[edge_type][1], dnids)),
                                         None)
                else:
                    graphs[edge_type] = (snids, dnids, None)

        # we assume edges have features is not loaded by edgeLoader.
        for feat_loader in self._feature_loader:
            if feat_loader.node_feat is False:
                # {edge_type: (snids, dnids, feats)}
                edge_feat_result = feat_loader.process(self._node_dict)
                edge_feat_results.append(edge_feat_result)

                for edge_type, vals in edge_feat_result.items():
                    feats = {}
                    snids, dnids, _ = vals[next(iter(vals.keys()))]

                    for feat_name, val in vals.items():
                        assert val[0].shape[0] == val[1].shape[0], \
                            'Edges with edge type {} has multiple features, ' \
                            'But some features do not cover all the edges.' \
                            'Expect {} edges, but get {} edges with edge feature {}.'.format(
                                edge_type if edge_type is not None else "",
                                snids.shape[0], val[0].shape[0], feat_name)
                        feats[feat_name] = val[2]

                    if edge_type in graphs:
                        new_feats = {}
                        for feat_name, feat in feats:
                            assert graphs[edge_type][2] is not None, \
                                'All edges under edge type {} should has features.' \
                                'Please check if you use EdgeLoader to load edges ' \
                                'for the same edge type'.format(
                                    edge_type if edge_type is not None else "")
                            assert feat_name not in graphs[edge_type][2], \
                                'Can not concatenate edges with features with other edges without features'
                            assert graphs[edge_type][2][feat_name].shape[1:] == feat.shape[1:], \
                                'Can not concatenate edges with different feature shape'

                            new_feats[feat_name] = np.concatenate((graphs[edge_type][2][feat_name], feat))

                        graphs[edge_type] = (np.concatenate((graphs[edge_type][0], snids)),
                                             np.concatenate((graphs[edge_type][1], dnids)),
                                             new_feats)
                    else:
                        graphs[edge_type] = (snids, dnids, feats)
            else:
                # {node_type: {feat_name :(node_ids, node_feats)}}
                node_feat_result = feat_loader.process(self._node_dict)
                node_feat_results.append(node_feat_result)

                for node_type, vals in node_feat_result.items():
                    nids, _ = vals[next(iter(vals.keys()))]
                    max_nid = int(np.max(nids)) + 1
                    if node_type in nodes:
                        nodes[node_type] = max(nodes[node_type]+1, max_nid)
                    else:
                        nodes[node_type] = max_nid

        for label_loader in self._label_loader:
            if label_loader.node_label is False:
                # {edge_type: ((train_snids, train_dnids, train_labels,
                #               valid_snids, valid_dnids, valid_labels,
                #               test_snids, test_dnids, test_labels)}
                if self._label_map is None:
                    edge_label_result = label_loader.process(self._node_dict)
                    self._label_map = label_loader.label_map
                else:
                    edge_label_result = label_loader.process(self._node_dict,
                                                             label_map=self._label_map)
                    for idx, label in self._label_map.items():
                        assert label == label_loader.label_map[idx], \
                            'All label files should have the same label set'
                edge_label_results.append(edge_label_result)

                for edge_type, vals in edge_label_result.items():
                    train_snids, train_dnids, train_labels, \
                        valid_snids, valid_dnids, valid_labels, \
                        test_snids, test_dnids, test_labels = vals

                    if edge_type in graphs:
                        # If same edge_type also has features,
                        # we expect edges have labels also have features.
                        # Thus we avoid add edges twice.
                        # Otherwise, if certain edge_type has no featus, add it directly
                        if graphs[edge_type][2] is None:
                            snids = graphs[edge_type][0]
                            dnids = graphs[edge_type][1]
                            if train_snids is not None:
                                snids = np.concatenate((snids, train_snids))
                                dnids = np.concatenate((dnids, train_dnids))
                            if valid_snids is not None:
                                snids = np.concatenate((snids, valid_snids))
                                dnids = np.concatenate((dnids, valid_dnids))
                            if test_snids is not None:
                                snids = np.concatenate((snids, test_snids))
                                dnids = np.concatenate((dnids, test_dnids))
                            graphs[edge_type] = (snids, dnids, None)
                    else:
                        snids = np.empty((0,), dtype='long')
                        dnids = np.empty((0,), dtype='long')
                        if train_snids is not None:
                            snids = np.concatenate((snids, train_snids))
                            dnids = np.concatenate((dnids, train_dnids))
                        if valid_snids is not None:
                            snids = np.concatenate((snids, valid_snids))
                            dnids = np.concatenate((dnids, valid_dnids))
                        if test_snids is not None:
                            snids = np.concatenate((snids, test_snids))
                            dnids = np.concatenate((dnids, test_dnids))
                        graphs[edge_type] = (snids, dnids, None)
            else:
                # {node_type: (train_nids, train_labels,
                #              valid_nids, valid_labels,
                #              test_nids, test_labels)}
                if self._label_map is None:
                    node_label_result = label_loader.process(self._node_dict)
                    self._label_map = label_loader.label_map
                else:
                    node_label_result = label_loader.process(self._node_dict,
                                                             label_map=self._label_map)
                    for idx, label in self._label_map.items():
                        assert label == label_loader.label_map[idx], \
                            'All label files should have the same label set'
                node_label_results.append(node_label_result)
                for node_type, vals in node_label_result.items():
                    train_nids, _, valid_nids, _, test_nids, _ = vals
                    max_nid = 0
                    if train_nids is not None:
                        max_nid = max(int(np.max(train_nids))+1, max_nid)
                    if valid_nids is not None:
                        max_nid = max(int(np.max(valid_nids))+1, max_nid)
                    if test_nids is not None:
                        max_nid = max(int(np.max(test_nids))+1, max_nid)

                    if node_type in nodes:
                        nodes[node_type] = max(nodes[node_type], max_nid)
                    else:
                        nodes[node_type] = max_nid
        if self._verbose:
            print('Done processing graph structure.')
            print('Start building dgl graph.')

        # build graph
        if len(graphs) > 1:
            assert None not in graphs, \
                'With heterogeneous graph, all edges should have edge type'
            assert None not in nodes, \
                'With heterogeneous graph, all nodes should have node type'
            graph_edges = {key: (val[0], val[1]) for key, val in graphs.items()}
            for etype, (src_nids, dst_nids) in graph_edges.items():
                src_max_nid = int(np.max(src_nids))+1
                dst_max_nid = int(np.max(dst_nids))+1
                if etype[0] in nodes:
                    nodes[etype[0]] = max(nodes[etype[0]], src_max_nid)
                else:
                    nodes[etype[0]] = src_max_nid
                if etype[2] in nodes:
                    nodes[etype[2]] = max(nodes[etype[2]], dst_max_nid)
                else:
                    nodes[etype[2]] = dst_max_nid

            g = dgl.heterograph(graph_edges, num_nodes_dict=nodes)
            for edge_type, vals in graphs.items():
                # has edge features
                if vals[2] is not None:
                    for key, feat in vals[2].items():
                        g.edges[edge_type].data[key] = th.tensor(feat)
        else:
            g = dgl.graph((graphs[None][0], graphs[None][1]), num_nodes=nodes[None])
            # has edge features
            if graphs[None][2] is not None:
                for key, feat in graphs[None][2].items():
                    g.edata[key] = th.tensor(feat)

        # no need to handle edge features
        # handle node features
        for node_feats in node_feat_results:
            # {node_type: (node_ids, node_feats)}
            for node_type, vals in node_feats.items():
                if node_type is None:
                    for key, feat in vals.items():
                        g.ndata[key] = th.tensor(feat[1])
                else:
                    for key, feat in vals.items():
                        g.nodes[node_type].data[key] = th.tensor(feat[1])

        if self._verbose:
            print('Done building dgl graph.')
            print('Start processing graph labels...')
        train_edge_labels = {}
        valid_edge_labels = {}
        test_edge_labels = {}
        # concatenate all edge labels
        for edge_label_result in edge_label_results:
            for edge_type, vals in edge_label_result.items():
                train_snids, train_dnids, train_labels, \
                    valid_snids, valid_dnids, valid_labels, \
                    test_snids, test_dnids, test_labels = vals

                # train edge labels
                if train_snids is not None:
                    if edge_type in train_edge_labels:
                        train_edge_labels[edge_type] = (
                            np.concatenate((train_edge_labels[edge_type][0], train_snids)),
                            np.concatenate((train_edge_labels[edge_type][1], train_dnids)),
                            None if train_labels is None else \
                                np.concatenate((train_edge_labels[edge_type][2], train_labels)))
                    else:
                        train_edge_labels[edge_type] = (train_snids, train_dnids, train_labels)

                # valid edge labels
                if valid_snids is not None:
                    if edge_type in valid_edge_labels:
                        valid_edge_labels[edge_type] = (
                            np.concatenate((valid_edge_labels[edge_type][0], valid_snids)),
                            np.concatenate((valid_edge_labels[edge_type][1], valid_dnids)),
                            None if valid_labels is None else \
                                np.concatenate((valid_edge_labels[edge_type][2], valid_labels)))
                    else:
                        valid_edge_labels[edge_type] = (valid_snids, valid_dnids, valid_labels)

                # test edge labels
                if test_snids is not None:
                    if edge_type in test_edge_labels:
                        test_edge_labels[edge_type] = (
                            np.concatenate((test_edge_labels[edge_type][0], test_snids)),
                            np.concatenate((test_edge_labels[edge_type][1], test_dnids)),
                            None if test_labels is None else \
                                np.concatenate((test_edge_labels[edge_type][2], test_labels)))
                    else:
                        test_edge_labels[edge_type] = (test_snids, test_dnids, test_labels)

        # create labels and train/valid/test mask
        assert len(train_edge_labels) >= len(valid_edge_labels), \
            'The training set should cover all kinds of edge types ' \
            'where the validation set is avaliable.'
        assert len(train_edge_labels) == len(test_edge_labels), \
            'The training set should cover the same edge types as the test set.'

        for edge_type, train_val in train_edge_labels.items():
            train_snids, train_dnids, train_labels = train_val
            if edge_type in valid_edge_labels:
                valid_snids, valid_dnids, valid_labels = valid_edge_labels[edge_type]
            else:
                valid_snids, valid_dnids, valid_labels = None, None, None
            assert edge_type in test_edge_labels
            test_snids, test_dnids, test_labels = test_edge_labels[edge_type]

            u, v, eids = g.edge_ids(train_snids,
                                    train_dnids,
                                    return_uv=True,
                                    etype=edge_type)
            labels = None
            # handle train label
            if train_labels is not None:
                assert train_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the training set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                train_labels = th.tensor(train_labels)
                labels = th.full((g.num_edges(edge_type), train_labels.shape[1]),
                                    -1,
                                    dtype=train_labels.dtype)
                labels[eids] = train_labels
            # handle train mask
            train_mask = th.full((g.num_edges(edge_type),), False, dtype=th.bool)
            train_mask[eids] = True

            valid_mask = None
            if valid_snids is not None:
                u, v, eids = g.edge_ids(valid_snids,
                                        valid_dnids,
                                        return_uv=True,
                                        etype=edge_type)
                assert valid_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the validation set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                # handle valid label
                if valid_labels is not None:
                    assert labels is not None, \
                        'We must have train_labels first then valid_labels'
                    labels[eids] = th.tensor(valid_labels)
                # handle valid mask
                valid_mask = th.full((g.num_edges(edge_type),), False, dtype=th.bool)
                valid_mask[eids] = True

            u, v, eids = g.edge_ids(test_snids,
                                    test_dnids,
                                    return_uv=True,
                                    etype=edge_type)
            # handle test label
            if test_labels is not None:
                assert test_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the testing set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                assert labels is not None, \
                    'We must have train_labels first then test_lavbels'
                labels[eids] = th.tensor(test_labels)
            # handle test mask
            test_mask = th.full((g.num_edges(edge_type),), False, dtype=th.bool)
            test_mask[eids] = True

            # add label and train/valid/test masks into g
            if edge_type is None:
                assert len(train_edge_labels) == 1, \
                    'Homogeneous graph only supports one type of labels'
                if labels is not None:
                    g.edata['labels'] = labels
                g.edata['train_mask'] = train_mask
                g.edata['valid_mask'] = valid_mask
                g.edata['test_mask'] = test_mask
            else: # we have edge type
                assert 'train_mask' not in g.edges[edge_type].data
                if labels is not None:
                    g.edges[edge_type].data['labels'] = labels
                g.edges[edge_type].data['train_mask'] = train_mask
                g.edges[edge_type].data['valid_mask'] = valid_mask
                g.edges[edge_type].data['test_mask'] = test_mask

        # node labels
        train_node_labels = {}
        valid_node_labels = {}
        test_node_labels = {}
        for node_labels in node_label_results:
            for node_type, vals in node_labels.items():
                train_nids, train_labels, \
                    valid_nids, valid_labels, \
                    test_nids, test_labels = vals

                # train node labels
                if train_nids is not None:
                    if node_type in train_node_labels:
                        train_node_labels[node_type] = (
                            np.concatenate((train_node_labels[node_type][0], train_nids)),
                            None if train_labels is None else \
                                np.concatenate((train_node_labels[node_type][1], train_labels)))
                    else:
                        train_node_labels[node_type] = (train_nids, train_labels)

                # valid node labels
                if valid_nids is not None:
                    if node_type in valid_node_labels:
                        valid_node_labels[node_type] = (
                            np.concatenate((valid_node_labels[node_type][0], valid_nids)),
                            None if valid_labels is None else \
                                np.concatenate((valid_node_labels[node_type][0], valid_labels)))
                    else:
                        valid_node_labels[node_type] = (valid_nids, valid_labels)

                # test node labels
                if test_nids is not None:
                    if node_type in test_node_labels:
                        test_node_labels[node_type] = (
                            np.concatenate((test_node_labels[node_type][0], test_nids)),
                            None if test_labels is none else \
                                np.concatenate((test_node_labels[node_type][0], test_labels)))
                    else:
                        test_node_labels[node_type] = (test_nids, test_labels)

        # create labels and train/valid/test mask
        assert len(train_node_labels) >= len(valid_node_labels), \
            'The training set should cover all kinds of node types ' \
            'where the validation set is avaliable.'
        assert len(train_node_labels) == len(test_node_labels), \
            'The training set should cover the same node types as the test set.'

        for node_type, train_val in train_node_labels.items():
            train_nids, train_labels = train_val
            if node_type in valid_node_labels:
                valid_nids, valid_labels = valid_node_labels[node_type]
            else:
                valid_nids, valid_labels = None, None
            test_nids, test_labels = test_node_labels[node_type]

            labels = None
            # handle train label
            if train_labels is not None:
                train_labels = th.tensor(train_labels)
                labels = th.full((g.num_nodes(node_type), train_labels.shape[1]),
                                 -1,
                                 dtype=train_labels.dtype)
                labels[train_nids] = train_labels
            # handle train mask
            train_mask = th.full((g.num_nodes(node_type),), False, dtype=th.bool)
            train_mask[train_nids] = True

            valid_mask = None
            if valid_nids is not None:
                # handle valid label
                if valid_labels is not None:
                    assert labels is not None, \
                        'We must have train_labels first then valid_labels'
                    labels[valid_nids] = th.tensor(valid_labels)
                # handle valid mask
                valid_mask = th.full((g.num_nodes(node_type),), False, dtype=th.bool)
                valid_mask[valid_nids] = True

            # handle test label
            if test_labels is not None:
                assert labels is not None, \
                    'We must have train_labels first then test_labels'
                labels[test_nids] = th.tensor(test_labels)
            # handle test mask
            test_mask = th.full((g.num_nodes(node_type),), False, dtype=th.bool)
            test_mask[test_nids] = True

            # add label and train/valid/test masks into g
            if node_type is None:
                assert len(train_node_labels) == 1, \
                    'Homogeneous graph only supports one type of labels'
                g.ndata['labels'] = labels
                g.ndata['train_mask'] = train_mask
                g.ndata['valid_mask'] = valid_mask
                g.ndata['test_mask'] = test_mask
            else: # we have node type
                assert 'train_mask' not in g.nodes[node_type].data
                g.nodes[node_type].data['labels'] = labels
                g.nodes[node_type].data['train_mask'] = train_mask
                g.nodes[node_type].data['valid_mask'] = valid_mask
                g.nodes[node_type].data['test_mask'] = test_mask

        if self._verbose:
            print('Done processing labels')

        self._g = g

    def save(self, path):
        """save the graph and the labels"""
        graph_path = os.path.join(path,
                                  'graph.bin')
        info_path = os.path.join(path,
                                 'info.pkl')
        save_graphs(graph_path, self._g)
        save_info(info_path, {'node_id_map': self._node_dict,
                              'label_map': self._label_map})

    def load(self, path):
        graph_path = os.path.join(path,
                                  'graph.bin')
        info_path = os.path.join(path,
                                 'info.pkl')
        graphs, _ = load_graphs(graph_path)
        self._g = graphs[0]
        info = load_info(str(info_path))
        self._node_dict = info['node_id_map']
        self._label_map = info['label_map']

    @property
    def node_2_id(self):
        """ Return mappings from raw node id/name to internal node id

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return self._node_dict

    @property
    def id_2_node(self):
        """ Return mappings from internal node id to raw node id/name

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return {node_type : {val:key for key, val in node_maps.items()} \
            for node_type, node_maps in self._node_dict.items()}


    @property
    def label_map(self):
        """ Return mapping from internal label id to original label

        Return
        ------
        dict:
            {type: {label id(int) : raw label(string/int)}}
        """
        return self._label_map

    @property
    def graph(self):
        """ Return processed graph
        """
        return self._g