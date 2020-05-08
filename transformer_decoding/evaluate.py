import argparse
import json
from pathlib import Path
import tqdm
from _collections import defaultdict

import numpy as np
import torch

from transformers import (modeling_utils,
                          BartTokenizer,
                          BartForConditionalGeneration,
                          BartConfig)

from transformer_decoding import decoding_utils
import transformer_decoding.log as log

from newsroom.analyze.rouge import ROUGE_L, ROUGE_N

logger = log.create_logger(__name__)


def print_mean(results, rouge_types):
    for rouge_type in rouge_types:
        precs = results[rouge_type]['p']
        recalls = results[rouge_type]['r']
        fscores = results[rouge_type]['f']
        p = round(np.mean(precs), 3)
        r = round(np.mean(recalls), 3)
        f = round(np.mean(fscores), 3)
        print(rouge_type, 'p:', p, 'r:', r, 'f:', f)


class BartSummarizerConfig:
    def __init__(self, args):
        """
        currently we use the model `bart-large-cnn`
        """
        self.model = BartForConditionalGeneration.from_pretrained('model_id')
        self.tokenizer = BartTokenizer.from_pretrained(args['model_id'])


class Summarizer:

    def __init__(self, config):
        self.model = config['model']
        self.tokenizer = config['tokenizer']

    # TODO: factor out score computation vs reduction(?) -- wait for usecase


#  we want to be able to ensemble
#  (1) different inputs, same model
#  (2) same inputs, different models
#  -- we should support arbitrary combinations of these, without too much
#   cruft from configuration
def summarize_articles(articles, args):
    """
    Ensembled summarization of a cluster of articles
    """
    model = args['model']
    tokenizer = args['tokenizer']
    decoding_hyperparams = {
        'max_length': args['max_length'],
        'num_beams': args['num_beams']
    }

    component_states = [decoding_utils.get_start_state(a, model, tokenizer, decoding_hyperparams)
                        for a in articles]
    ensemble_state = decoding_utils.get_start_state(articles[0], model, tokenizer, decoding_hyperparams)

    component_states, ensemble_state = \
        decoding_utils.generate(component_states, decoding_hyperparams['max_length'],
                                ensemble_state=ensemble_state)

    # assert len(ensemble_state['input_ids']) == 1, 'We currently have batch size=1 (we decode one cluster at a time)'
    predictions = [tokenizer.decode(input_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for input_ids in ensemble_state['input_ids']]

    # TODO: currently `summaries` contains `beam_size` predictions, not sorted by score
    return predictions


def article_to_text(article, separator_token=' '):
    # just be sure about whitespace
    title = ' '.join(article["title"].strip().split())
    text = ' '.join(article["text"].strip().split())
    return f'{title} {separator_token} {text}'


def main(args):

    # TODO: all args in CLI
    hardcoded_args = {
        'model_id': 'bart-large-cnn',
        'max_length': 40,
        'num_beams': 3,
    }
    args = dict(hardcoded_args, **args)

    # load pretrained or finetuned transformer model

    print(f'loading pre-trained model: {args["model_id"]}')

    # fine-tuned
    if args['model_id'].endswith('.ckpt'):
    	from transformer_decoding.finetune import SummarizationTrainer
    	lightning_model = SummarizationTrainer.load_from_checkpoint(args['model_id'])
    	args['model'] = lightning_model.model
    	args['tokenizer'] = lightning_model.tokenizer
    else:
        # transformers pretrained
        args['model'] = BartForConditionalGeneration.from_pretrained(args['model_id'])
        args['tokenizer'] = BartTokenizer.from_pretrained(args['model_id'])

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    args['model'].eval()

    # WORKING: we have to load fine-tuned models in a different way because of pytorch-lightning

    if torch.cuda.is_available():
        args['model'].to('cuda')

    # summarize MDS / summarization dataset with model

    # print and write out evaluation results
    # TODO: WORKING: in general we want to be able to ensemble both models _and_
    dataset = [json.loads(l) for l in open(args['evaluation_dataset'])][:args['rows_to_eval']]

    # WORKING: also write out summaries as they're generated
    preds_output = open('eval_predicted_summaries.out', 'w', buffering=1)
    gold_output = open('eval_gold_summaries.out', 'w', buffering=1)

    summaries = []
    # get summary for each cluster
    # note here we have a macro-batch size of one cluster by definition
    for cluster in tqdm.tqdm(dataset):
        articles = [article_to_text(a) for a in cluster['articles'][:args['max_articles_in_cluster']]]

        predictions = summarize_articles(articles, args)
        #print(f'Predictions: \n{predictions}')
        #print()
        #print(f'input_ids shape: {ensemble_state["input_ids"].shape}')
        #print(f'Reference Summary:\n{cluster["summary"]}')

        # NOTE: hack to just take the first one right now, disregarding scores of different beam items
        predicted_summary = predictions[0]
        gold_summary = cluster['summary']
        summaries.append((predicted_summary, gold_summary))
        preds_output.write(f'{predicted_summary}\n')
        gold_output.write(f'{gold_summary}\n')

    preds_output.close()
    gold_output.close()

    # WORKING: get summary, append (or eval online)
    # Now evaluate
    rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']
    results = dict((rouge_type, defaultdict(list))
                   for rouge_type in rouge_types)

    # TODO: move to config args
    lowercase = True

    for hyp, ref in summaries:
        if lowercase:
            hyp = hyp.lower()
            ref = ref.lower()

        r1 = ROUGE_N(ref, hyp, n=1)
        r2 = ROUGE_N(ref, hyp, n=2)
        rl = ROUGE_L(ref, hyp)

        for (rouge_type, scores) in zip(rouge_types, [r1, r2, rl]):
            results[rouge_type]['p'].append(scores.precision)
            results[rouge_type]['r'].append(scores.recall)
            results[rouge_type]['f'].append(scores.fscore)

    print_mean(results, rouge_types)


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    '--resource-path',
    #    type=str,
    #    required=True,
    #    help='path to resources'
    #)
    parser.add_argument(
        '--evaluation-dataset',
        type=str,
        required=True,
        help='filepath of evaluation data'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='(currently) the model id string from the huggingface transformers library'
    )
    parser.add_argument(
        '--max-articles-in-cluster',
        type=int,
        required=False,
        default=None,
        help='take the first K articles in each cluster to use in the ensemble'
    )
    parser.add_argument(
        '--rows-to-eval',
        type=int,
        required=False,
        default=None,
        help='if provided, truncate eval dataset to this many rows'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(vars(parse_args()))
