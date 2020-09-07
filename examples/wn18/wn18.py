import os

from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, extract_archive
import dgl_graphloader

class WN18Dataset(DGLDataset):
    """ Example of loading wn18
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = _get_dgl_url('dataset/wn18.tgz')
        super(WN18Dataset, self).__init__('wn18',
                                          url=url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def download(self):
        tgz_path = os.path.join(self.raw_dir, 'wn18.tgz')
        download(self.url, path=tgz_path)
        extract_archive(tgz_path, self.raw_path)

    def process(self):
        root_path = self.raw_path
        train_path = os.path.join(root_path, 'train.txt')
        valid_path = os.path.join(root_path, 'valid.txt')
        test_path = os.path.join(root_path, 'test.txt')

        train_edge_label_loader = dgl_graphloader.EdgeLabelLoader(train_path)
        train_edge_label_loader.addRelationalTrainSet([0,2,1], src_node_type='n', dst_node_type='n')
        valid_edge_label_loader = dgl_graphloader.EdgeLabelLoader(valid_path)
        valid_edge_label_loader.addRelationalValidSet([0,2,1], src_node_type='n', dst_node_type='n')
        test_edge_label_loader = dgl_graphloader.EdgeLabelLoader(test_path)
        test_edge_label_loader.addRelationalTestSet([0,2,1], src_node_type='n', dst_node_type='n')

        graphloader = dgl_graphloader.GraphLoader(name='example')
        graphloader.appendLabel(train_edge_label_loader)
        graphloader.appendLabel(valid_edge_label_loader)
        graphloader.appendLabel(test_edge_label_loader)
        graphloader.addReverseEdge()
        graphloader.process()

        self._graphloader = graphloader

    def save(self):
        self._graphloader.save(self.raw_path)

    @property
    def g(self):
        return self._graphloader.graph

if __name__ == '__main__':
    dataset = WN18Dataset(raw_dir='./data/')
    g = dataset.g
    print(g)
