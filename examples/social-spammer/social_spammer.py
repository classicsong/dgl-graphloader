import os

from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, extract_archive
import dgl_graphloader

class SSpmDataset(DGLDataset):
    """ Example of loading social_spammer
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        super(SSpmDataset, self).__init__('social_spammer',
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        root_path = self.raw_dir
        train_path = os.path.join(root_path, 'train.csv')
        valid_path = os.path.join(root_path, 'valid.csv')
        test_path = os.path.join(root_path, 'test.csv')
        train_node_label_loader = dgl_graphloader.NodeLabelLoader(train_path)
        train_node_label_loader.addTrainSet([0,1], node_type='user')
        valid_node_label_loader = dgl_graphloader.NodeLabelLoader(valid_path)
        valid_node_label_loader.addValidSet([0,1], node_type='user')
        test_node_label_loader = dgl_graphloader.NodeLabelLoader(test_path)
        test_node_label_loader.addTestSet([0,1], node_type='user')

        user_feat_path = os.path.join(root_path, 'usersdata.csv')
        rel_graph_path = os.path.join(root_path, 'relations.csv')
        user_feat_loader = dgl_graphloader.NodeFeatureLoader(user_feat_path)
        user_feat_loader.addCategoryFeature(["ID", "sex"], node_type='user')
        user_feat_loader.addNumericalBucketFeature(["ID", "age"],
                                                   range=[0,100],
                                                   bucket_cnt=10,
                                                   slide_window_size=5,
                                                   norm=None,
                                                   node_type='user')
        rel_edge_loader = dgl_graphloader.EdgeLoader(rel_graph_path)
        rel_edge_loader.addCategoryRelationEdge([2,3,4], src_type='user', dst_type='user')

        graphloader = dgl_graphloader.GraphLoader(name='example', verbose=True)
        graphloader.appendLabel(train_node_label_loader) 
        graphloader.appendLabel(valid_node_label_loader)
        graphloader.appendLabel(test_node_label_loader)
        graphloader.appendFeature(user_feat_loader)
        graphloader.appendEdge(rel_edge_loader)
        graphloader.addReverseEdge()
        graphloader.process()

        self._graphloader = graphloader

    def save(self):
        self._graphloader.save(self.raw_dir)

    @property
    def g(self):
        return self._graphloader.graph

if __name__ == '__main__':
    dataset = SSpmDataset(raw_dir='./social-spammer/')
    g = dataset.g
    print(g)
    print(g.nodes['user'].data['nf'].dtype)
