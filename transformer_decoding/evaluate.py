import argparse
import json
from pathlib import Path
import tqdm
import os
import shutil
from _collections import defaultdict

import numpy as np
import torch
import spacy

import pyrouge
import logging

from transformers import (modeling_utils,
                          BartTokenizer,
                          BartForConditionalGeneration,
                          BartConfig)

from transformer_decoding import decoding_utils
import transformer_decoding.log as log

from newsroom.analyze.rouge import ROUGE_L, ROUGE_N

logger = log.create_logger(__name__)


# BEGIN: utils for Lebanoff 2018 rouge
def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    #   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_args = ['-e', r._data_dir,
                  '-c',
                  '95',
                  '-2', '4',  # This is the only one we changed (changed the max skip from -1 to 4)
                  '-U',
                  '-r', '1000',
                  '-n', '4',
                  '-w', '1.2',
                  '-a',
                  '-l', '100']
    rouge_args = ' '.join(rouge_args)
    rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
    return r.output_to_dict(rouge_results)


# def rouge_log(results_dict, dir_to_write):
def rouge_log(results_dict):
    """Log ROUGE results to screen and write to file.
    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l", "s4", "su4"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    logging.info(log_str)  # log to screen
    #     results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    #     print("Writing final ROUGE results to %s...", results_file)
    #     with open(results_file, "w") as f:
    #         f.write(log_str)
    return log_str


def write_for_rouge(all_reference_sents, decoded_words, ex_index, rouge_dec_dir, rouge_ref_dir, nlp):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.
    Args:
        all_reference_sents: list of list of strings
        decoded_words: list of strings
        ex_index: int, the index with which to label the files
    """

    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:  # there is text remaining that doesn't end in "."
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    # note sentence splitting here
    all_reference_sents = [
        [make_html_safe(' '.join([str(w) for w in s])) for s in nlp(abstract).sents]
        for abstract in all_reference_sents
    ]

    # Write to file
    decoded_file = os.path.join(rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    for abs_idx, abs in enumerate(all_reference_sents):
        ref_file = os.path.join(rouge_ref_dir, "%06d_reference.%s.txt" % (
            ex_index, chr(ord('A') + abs_idx)))
        with open(ref_file, "w") as f:
            # one long line
            # f.write(' '.join(abs).lower() + '\n')

            # one sentence on each line
            for idx, sent in enumerate(abs):
                f.write(sent + "\n")

                # f.write(sent) if idx==len(abs)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
        # one long line
        # f.write(' '.join(decoded_sents).lower() + '\n')
        for idx, sent in enumerate(decoded_sents):
            f.write(sent + "\n")


def lebanoff_2018_rouge(system_hyp_file, evaluation_dataset):
    TEMP_EVAL_DIR = Path('rouge_evaluation_tempdir')
    rouge_dec_dir = TEMP_EVAL_DIR / 'rouge_dec_dir'
    rouge_ref_dir = TEMP_EVAL_DIR / 'rouge_ref_dir'
    rouge_dec_dir.mkdir(parents=True, exist_ok=True)
    rouge_ref_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    # dataset needs to be in .jsonl
    dataset_rows = [json.loads(l) for l in open(evaluation_dataset)]

    # tokenize hyps to follow Lebanoff et al 2018 logic
    system_hyp_tokens = [[str(t) for t in nlp(h.strip())] for h in open(system_hyp_file)]

    # write the rouge files
    for idx, (row, h) in enumerate(zip(dataset_rows, system_hyp_tokens)):
        if type(row['summary']) is list:
            summaries = row['summary']
        else:
            summaries = [row['summary']]
        #     print(f'{len(summaries)} summaries available at row {idx}')
        write_for_rouge(summaries, h, idx, rouge_dec_dir, rouge_ref_dir, nlp)

    log_report = rouge_log(rouge_eval(rouge_ref_dir, rouge_dec_dir))
    print(log_report)
    shutil.rmtree(TEMP_EVAL_DIR)

# END: utils for Lebanoff 2018 rouge


def print_mean(results, rouge_types):
    for rouge_type in rouge_types:
        precs = results[rouge_type]['p']
        recalls = results[rouge_type]['r']
        fscores = results[rouge_type]['f']
        p = round(np.mean(precs), 3)
        r = round(np.mean(recalls), 3)
        f = round(np.mean(fscores), 3)
        print(rouge_type, 'p:', p, 'r:', r, 'f:', f)


def evaluate_rouge(hyps, refs, lowercase=True):
    # WORKING: get summary, append (or eval online)
    # Now evaluate
    rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']
    results = dict((rouge_type, defaultdict(list))
                   for rouge_type in rouge_types)

    for hyp, ref in zip(hyps, refs):
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

    return results, rouge_types


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
        'max_length': args['max_src_length'],
        'max_tgt_length': args['max_tgt_length'],
        'num_beams': args['num_beams']
    }

    component_states = [decoding_utils.get_start_state(a, model, tokenizer, decoding_hyperparams)
                        for a in articles]
    ensemble_state = decoding_utils.get_start_state(articles[0], model, tokenizer, decoding_hyperparams)

    component_states, ensemble_state = \
        decoding_utils.generate(component_states, decoding_hyperparams['max_tgt_length'],
                                ensemble_state=ensemble_state)

    # WORKING HERE: make sure predictions are sorted by score
    # TODO: this logic might move to end of `generate` function(?)
    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(ensemble_state['batch_size']):
        if ensemble_state['done'][batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        # Chris: TODO: we don't currently store `next_tokens` on state at each decoding step
        #if ensemble_state['eos_token_id'] is not None and all(
        #        (token_id % ensemble_state['vocab_size']).item() is not ensemble_state['eos_token_id'] for token_id in ensemble_state['next_tokens'][batch_idx]
        #):
        #    assert torch.all(
        #        ensemble_state['next_scores'][batch_idx, :ensemble_state['num_beams']] == ensemble_state['beam_scores'].view(ensemble_state['batch_size'], ensemble_state['num_beams'])[batch_idx]
        #    ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
        #        ensemble_state['next_scores'][:, :ensemble_state['num_beams']][batch_idx], ensemble_state['beam_scores'].view(ensemble_state['batch_size'], ensemble_state['num_beams'])[batch_idx],
        #    )

        # TODO: finished hyps (those that are `done`) should already be in `generated_hyps`, assert this is the case
        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(ensemble_state['num_beams']):
            effective_beam_id = batch_idx * ensemble_state['num_beams'] + beam_id
            final_score = ensemble_state['beam_scores'][effective_beam_id].item()
            final_tokens = ensemble_state['input_ids'][effective_beam_id]
            ensemble_state['generated_hyps'][batch_idx].add(final_tokens, final_score)


    #assert len(ensemble_state['input_ids']) == 1, 'We currently have batch size=1 (we decode one cluster at a time)'
    assert ensemble_state['batch_size'] == 1, 'current logic assumes batch size = 1'

    # sort hyps by score (0 index is first batch, and we're assuming batch_size always = 1 right now)
    sorted_hyps = [(hyp, score) for score, hyp in sorted(ensemble_state['generated_hyps'][0].beams, key=lambda b: b[0], reverse=True)]

    #predictions = [tokenizer.decode(input_ids,
    #                                skip_special_tokens=True,
    #                                clean_up_tokenization_spaces=False)
    #               for input_ids in ensemble_state['input_ids']]

    predictions = [tokenizer.decode(hyp,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for hyp, _ in sorted_hyps]

    import ipdb; ipdb.set_trace()

    # TODO: currently `summaries` contains `beam_size` predictions, not sorted by score
    return predictions


def article_to_text(article, separator_token=' '):
    # just be sure about whitespace
    title = ' '.join(article["title"].strip().split())
    text = ' '.join(article["text"].strip().split())
    return f'{title} {separator_token} {text}'


def main(args):

    # load pretrained or finetuned transformer model
    print(f'loading pre-trained model: {args["model_id"]}')

    # we have to load fine-tuned models in a different way because of pytorch-lightning
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

    if torch.cuda.is_available():
        args['model'].to('cuda')

    # summarize MDS / summarization dataset with model

    # print and write out evaluation results
    if args['evaluation_dataset'].endswith('.jsonl'):
        dataset = [json.loads(l) for l in open(args['evaluation_dataset'])][:args['rows_to_eval']]
    else:
        raise AssertionError('Right now we only know how to handle .jsonl evaluation datasets')

    # WORKING: also write out summaries as they're generated
    preds_output = open('eval_predicted_summaries.out', 'w', buffering=1)
    gold_output = open('eval_gold_summaries.out', 'w', buffering=1)

    summaries = []
    # get summary for each cluster
    # note here we have a macro-batch size of one cluster by definition
    for cluster in tqdm.tqdm(dataset):
        articles = [article_to_text(a) for a in cluster['articles'][:args['max_articles_in_cluster']]]
        if args['min_input_char_length'] is not None:
            articles_ = [a for a in articles if len(a) >= args['min_input_char_length']]
            if len(articles_) == 0:
                articles_ = [articles[0]]
            articles = articles_

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

    # Evaluation
    hyps, refs = zip(*summaries)
    results, rouge_types = evaluate_rouge(hyps, refs)

    print_mean(results, rouge_types)
    # End evaluation


# TODO: one arg set for evaluation on files, one for predicting, writing files, and evaluating with model
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
        help='the model id string from the huggingface transformers library, or the path to a pytorch lightning fine-tuned .ckpt'
    )
    parser.add_argument(
        '--num-beams',
        type=int,
        required=False,
        default=1,
        help='number of beam search beams'
    )
    parser.add_argument(
        '--min-input-char-length',
        type=int,
        required=False,
        default=None,
        help='If specified, we will try to filter inputs to be at least this many characters'
    )
    parser.add_argument(
        '--max-src-length',
        type=int,
        required=False,
        default=256,
        help='The maximum length of input sequences'
    )
    parser.add_argument(
        '--max-tgt-length',
        type=int,
        required=False,
        default=64,
        help='The maximum length of decoded sequences'
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
