#### fine-tuning on multi-doc summarization datasets
(1) flatten dataset (currently see notebooks) to cnn-dm *.sources and *.targets format
(2) run `bin/run_train.sh`
```
pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/

# - '' is a correct model identifier listed on 'https://huggingface.co/models'

# - or '' is the correct path to a directory containing a 'config.json' file

```

5.5.20 -- for discussion, results of dynamic ensembling on WCEP test.jsonl with 1 article vs 5 articles:
```
RUN_FLAGS='--max-articles-in-cluster 1' make evaluate
rouge-1 p: 0.286 r: 0.296 f: 0.275
rouge-2 p: 0.085 r: 0.092 f: 0.083
rouge-l p: 0.206 r: 0.218 f: 0.201
RUN_FLAGS='--max-articles-in-cluster 5' make evaluate
rouge-1 p: 0.311 r: 0.327 f: 0.303
rouge-2 p: 0.098 r: 0.108 f: 0.097
rouge-l p: 0.227 r: 0.244 f: 0.223
```
