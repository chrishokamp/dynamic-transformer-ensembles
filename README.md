## DynE: Dynamic Ensemble Decoding for Multi-Document Summarization


## :star: Complete README coming soon :star:


### Installation



### Preparing a dataset

(1) flatten dataset (currently see notebooks) to cnn-dm *.sources and *.targets format
(2) run `bin/run_train.sh`
```
pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/

# - '' is a correct model identifier listed on 'https://huggingface.co/models'

# - or '' is the correct path to a directory containing a 'config.json' file


Datasets already processed in required format:
WCEP
MultiNews
DUC2004

```
### Fine-tuning 


#### Fine-Tuned Model Checkpoints


### Evaluation
```

```


If you use ideas or code from this project, please cite:
```
@article{DynamicEnsembles,
    title = {DynE: Dynamic Ensemble Decoding for Multi-Document Summarization},
    author = {Chris Hokamp and Demian Gholipour Ghalandari and Nghia The Pham
              and John Glover},
    journal={arXiv preprint arXiv:},
    year = {2020},
}



```
