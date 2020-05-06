#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# flatten a multidoc summarization dataset in .jsonl format to a parallel dataset that uses
# the *.sources *.targets format from cnn-dm

# TODO: support shuffling since the cluster items will be sequential by default

# TODO: support formatting with special tokens to indicate document structure (i.e. <SEP> token between Title and Body)



# In[1]:


from pathlib import Path
import json
import tqdm

import numpy as np

from transformer_decoding.evaluate import article_to_text


DATADIR = Path('/home/chris/projects/aylien/dynamic-ensembles/data/WCEP')
prefixes = ['train', 'val']
shuffle = True
separator_token = ' [SEP] '


for dataset_prefix in prefixes:
    sources_and_targets = []
    cluster_cnt = 0
    print('loading clusters')
    for cluster in tqdm.tqdm((json.loads(l) for l in open(DATADIR / (dataset_prefix + '.jsonl')))):
        for article in cluster['articles']:
            sources_and_targets.append((article_to_text(article, separator_token=separator_token), cluster['summary']))
        cluster_cnt += 1
    
    output_idxs = np.arange(len(sources_and_targets))
    if shuffle:
        np.random.shuffle(output_idxs)
    
    with open(DATADIR / (dataset_prefix + '.sources'), 'w') as srcs, open(DATADIR / (dataset_prefix + '.targets'), 'w') as tgts:
        for idx in tqdm.tqdm(output_idxs):
            src = sources_and_targets[idx][0]
            tgt = sources_and_targets[idx][1]
            srcs.write(f'{src}\n')
            tgts.write(f'{tgt}\n')
    print(f'wrote {len(sources_and_targets)} segments from {cluster_cnt} clusters to {srcs.name} and {tgts.name}')
            
        
    


# In[ ]:




