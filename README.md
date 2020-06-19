## DynE: Dynamic Ensemble Decoding for Multi-Document Summarization

This repo contains the code for [DynE: Dynamic Ensemble Decoding for Multi-Document Summarization](https://arxiv.org/abs/2006.08748).

This code base can be used to add dynamic ensembling capability to models from the [Huggingface transformers library](https://github.com/huggingface/transformers).

## :star: Complete README coming soon :star:

### Multi-Document Summarization (MDS) Datasets

MDS datasets in the format required by the scripts in this repo:
- [WCEP](https://drive.google.com/drive/folders/1KSxlIx9Hq6l3pTTvsrbug-gpeuQIrQgW?usp=sharing) (train, val, test)
- [MultiNews](https://drive.google.com/drive/folders/1nuBM8aMjauA7bKOdPeQf6DeiR8-TeMaR?usp=sharing) (train, val, test)
- [DUC2004](https://drive.google.com/drive/folders/1q11LDSGqan-zHiMgA8IiB-vnfIXz39IJ?usp=sharing) (test)

### Model Checkpoints and Outputs

##### Model Checkpoints

We fine-tune the `bart-large-cnn` single-document summarization model from the [transformers library](https://github.com/huggingface/transformers)
- The best fine-tuned model checkpoints WCEP and MultiNews are [here](https://drive.google.com/drive/folders/1dCwg-sd0bPiZZV7nDLOO2ZoUcCDRiO3V?usp=sharing)

##### Fine-tuned Model Outputs

- Download the outputs of fine-tuned models on the test sets of WCEP and MultiNews [here](https://drive.google.com/drive/folders/1dCwg-sd0bPiZZV7nDLOO2ZoUcCDRiO3V?usp=sharing)


If you use ideas or code from this project, please cite:
```
@article{DynamicEnsembles,
    title = {DynE: Dynamic Ensemble Decoding for Multi-Document Summarization},
    author = {Chris Hokamp and Demian Gholipour Ghalandari and Nghia The Pham
              and John Glover},
    journal={arXiv preprint arXiv:2006.08748},
    year = {2020},
}

```

