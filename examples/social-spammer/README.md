# Social Spammer dataset
The dataset (user feature data) is cleared up to ignore unlabeled nodes.
The dataset an be slightly different from the https://linqs-data.soe.ucsc.edu/public/social_spammer/ and we do a random partition of train/valid/test.
Please refer to https://dgl-neptune-example.s3.amazonaws.com/social-pammers/preprocess.py for how data is processed.

## Loading dataset into DGLGraph
```
>>> wget https://dgl-neptune-example.s3.amazonaws.com/social-pammers/social-spammer.tgz
>>> tar zxf social-spammer.tgz
>>> python3 social_spammer.py
``` 
