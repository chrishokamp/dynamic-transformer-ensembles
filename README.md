## DynE: Dynamic Ensemble Decoding for Multi-Document Summarization

This repo contains the code for [DynE: Dynamic Ensemble Decoding for Multi-Document Summarization](https://arxiv.org/abs/2006.08748).

This code base can be used to add dynamic ensembling capability to models from the [Huggingface transformers library](https://github.com/huggingface/transformers).

## :star: Complete README coming soon :star:

### Multi-Document Summarization (MDS) Datasets

MDS datasets in the format required by the scripts in this repo:
- [WCEP](https://drive.google.com/drive/folders/1KSxlIx9Hq6l3pTTvsrbug-gpeuQIrQgW?usp=sharing) (train, val, test)
- [MultiNews](https://drive.google.com/drive/folders/1nuBM8aMjauA7bKOdPeQf6DeiR8-TeMaR?usp=sharing) (train, val, test)
- [DUC2004](https://drive.google.com/drive/folders/1q11LDSGqan-zHiMgA8IiB-vnfIXz39IJ?usp=sharing) (test)

The original WCEP dataset used to generate the flat training data:
- [WCEP in `.jsonl` format](https://drive.google.com/drive/folders/1PJufMEOdogIaKQq-PlB4vvawLa6tvEG6)

----------------------

### Model Checkpoints and Outputs

##### Model Checkpoints

We fine-tune the `bart-large-cnn` single-document summarization model from the [transformers library](https://github.com/huggingface/transformers)
- The best fine-tuned model checkpoints for WCEP and MultiNews are [here](https://drive.google.com/drive/folders/1B449P6kwm6_6AjpaASduGMi3Ff6Z1IBd?usp=sharing)

##### Fine-tuned Model Outputs

- Download the outputs of fine-tuned models on the test sets of WCEP and MultiNews [here](https://drive.google.com/drive/folders/1dCwg-sd0bPiZZV7nDLOO2ZoUcCDRiO3V?usp=sharing)

----------------------

### Evaluation
Prediction and evaluation are done by the script `transformer_decoding/evaluate.py`
There is also a `make` task for evaluation which simply calls this script.

For example, to predict using a model id from `transformers`, or with a fine-tuned model checkpoint,
and evaluate with the Ghalandari et al. 2020 evaluation workflow:
```
MODEL_ID=model_checkpoints/wcep_fine-tune-bart-large/checkpointepoch\=1.ckpt \
RUN_FLAGS='--max-articles-in-cluster 5 --max-src-length 512 --max-tgt-length 64 --num-beams 5 --eval-prefix wcep_5_articles_' \
make evaluate
```
- pretrained model checkpoints can be downloaded from the links above. 

For a quick test, use the `--rows-to-eval` argument, which will only predict the first `N` rows from the dataset:
```
MODEL_ID=model_checkpoints/wcep_fine-tune-bart-large/checkpointepoch\=1.ckpt \
RUN_FLAGS='--max-articles-in-cluster 5 --max-src-length 512 --max-tgt-length 64 --num-beams 5 --rows-to-eval 10 --eval-prefix wcep_5_articles_' \
make evaluate
```

To run evaluation only, using previously generated predictions, supply the `--predictions` argument to `transformer_decoding/evaluate.py`:
```
EVALUATION_DATASET=data/WCEP/test.jsonl \
RUN_FLAGS='--predictions outputs/wcep/wcep_5_articles_eval_predicted_summaries.out' \
make evaluate
```

##### Scoring Gold Summaries by Forced Decoding

```

EVALUATION_DATASET=data/WCEP/test.jsonl \
RUN_FLAGS='--force-decode-gold --max-articles-in-cluster 5 --max-src-length 512 --max-tgt-length 512 --num-beams 1 --rows-to-eval 10 --eval-prefix wcep_5_articles_' \
make evaluate

```

----------------------

### Citing

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

