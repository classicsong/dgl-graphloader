.. _apidgl_graphloader:

dgl_graphloader
==================

.. currentmodule:: dgl_graphloader
.. automodule:: dgl_graphloader

Loading Data From CSV
---------------------------

Operators for loading data from CSV files.

Feature loader
```````````````````````````````````
.. autoclass:: NodeFeatureLoader
    :members: addCategoryFeature, addMultiCategoryFeature, addNumericalFeature, addMultiNumericalFeature, addNumericalBucketFeature, addWord2VecFeature

.. autoclass:: EdgeFeatureLoader
    :members: addNumericalFeature

Graph structure loader
```````````````````````````````````
.. autoclass:: EdgeLoader
    :members: addEdges, addCategoryRelationEdge

Label loader
```````````````````````````````````
.. autoclass:: NodeLabelLoader
    :members: addTrainSet, addValidSet, addTestSet, addSet

.. autoclass:: EdgeLabelLoader
    :members: addTrainSet, addValidSet, addTestSet, addSet, addRelationalTrainSet, addRelationalValidSet, addRelationalTestSet, addRelationalSet

DGL Graph loader
---------------------------

.. autoclass:: GraphLoader
    :members: appendEdge, appendFeature, appendLabel, addReverseEdge, process, save, load