import argparse
import json
from pathlib import Path
import tqdm
import os
import shutil
import time
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


np.random.seed(42)


# BEGIN: utils for Lebanoff 2018 rouge
# adapted from here: 
# https://github.com/ucfnlp/multidoc_summarization/blob/ae30c9ee039d4ad5ff64fd2245faafc5a62c4dd7/decode.py

# installing pyrouge
# https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu
def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
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

# BEGIN: utils for Fabbri 2019 rouge
# Evaluation from: https://github.com/Alex-Fabbri/Multi-News/blob/3675e7c422ae3b4020617a324ac264f50333357d/code/OpenNMT-py-baselines/tools/test_rouge.py
def test_rouge(candidates, references):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
#         candidates = [line.strip() for line in cand]
#         references = [line.strip() for line in ref]
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = 'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def rouge_results_to_str(results_dict):
    return ">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100)

# END: utils for Fabbri 2019 rouge


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
    if type(hyps) is str:
        hyps = [l.strip() for l in open(hyps)]
    if type(refs) is str:
        assert refs.endswith('.jsonl'), 'reference summaries must be stored in "summaries": [] field of .jsonl file'
        refs = [json.loads(c)['summary'].strip() for c in open(refs['evaluation_dataset'])]

    assert len(hyps) == len(refs)
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
    # NOTE: could factor out score computation vs reduction(?) -- wait for usecase


#  eventually we want to be able to ensemble
#  (1) different inputs, same model
#  (2) same inputs, different models
#  -- we should support arbitrary combinations of these, without too much
#   cruft from configuration

def summarize_articles(articles, args, gold_summary=None):
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

    # TODO: WORKING: add flag in ensemble state to let user force decode

    component_states = [decoding_utils.get_start_state(a, model, tokenizer, decoding_hyperparams)
                        for a in articles]

    # Note we just pass the first article in cluster when building the ensemble state
    ensemble_state = decoding_utils.get_start_state(articles[0], model, tokenizer, decoding_hyperparams)

    # ((batch) x |vocab| x timesteps)
    timestep_mask = None
    if args['force_decode_gold']:
        # convert text to tensor
        # Note currently hard-coded max gold summary length
        encoded_gold = tokenizer.batch_encode_plus(
            [gold_summary],
            max_length=512,
            pad_to_max_length=False,
            return_tensors='pt'
        )
        gold_ids = encoded_gold['input_ids']

        # (timesteps, |vocab|)
        # set everything not in`float("inf")`
        # Note: since the mask is going to be elementwise-multiplied with logprobs, we set to float("inf") instead of
        # -float("inf") so that the sign doesn't get flipped
        # effectively we know our mask tensor for each timestep is (1, |vocab_size|),
        # because batch size and beam size are 1
        timestep_mask = torch.empty(gold_ids.shape[1], ensemble_state['vocab_size']).fill_(float("inf"))
        timestep_mask = timestep_mask.scatter(-1, gold_ids.T, 1.)[:, None, :]

    # WORKING TODO: attach gold summary to ensemble state if user wants to force decode
    # WORKING TODO: assert decoding hyperparams make sense if force-decoding (beam size = 1, etc...)

    component_states, ensemble_state = \
        decoding_utils.generate(component_states, decoding_hyperparams['max_tgt_length'],
                                ensemble_state=ensemble_state, timestep_mask=timestep_mask)

    # NOTE: this logic might move to end of `generate` function(?)
    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(ensemble_state['batch_size']):
        if ensemble_state['done'][batch_idx]:
            continue

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(ensemble_state['num_beams']):
            effective_beam_id = batch_idx * ensemble_state['num_beams'] + beam_id
            final_score = ensemble_state['beam_scores'][effective_beam_id].item()
            final_tokens = ensemble_state['input_ids'][effective_beam_id]

            hyp_metadata = []
            for state_idx in range(len(ensemble_state['decoding_stats'])):
                hyp_metadata.append(ensemble_state['decoding_stats'][state_idx][effective_beam_id])

            ensemble_state['generated_hyps'][batch_idx].add(final_tokens, final_score, metadata=hyp_metadata)

    assert ensemble_state['batch_size'] == 1, 'current logic assumes batch size = 1'

    # sort hyps by score (0 index is first batch, and we're assuming batch_size always = 1 right now)
    sorted_hyps = [(hyp, score, metadata) for score, hyp, metadata in sorted(ensemble_state['generated_hyps'][0].beams, key=lambda b: b[0], reverse=True)]

    print(f'Num hyps in BeamHypotheses: {len(sorted_hyps)}')

    # map token indexes back to strings
    predictions = [tokenizer.decode(hyp,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for hyp, _, _ in sorted_hyps]

    return predictions, sorted_hyps


def article_to_text(article, separator_token=' '):
    # just be sure about whitespace
    title = ' '.join(article["title"].strip().split())
    text = ' '.join(article["text"].strip().split())
    return f'{title} {separator_token} {text}'


def main(args):

    if args['evaluation_dataset'].endswith('.jsonl'):
        dataset = [json.loads(l) for l in open(args['evaluation_dataset'])][:args['rows_to_eval']]
    else:
        raise AssertionError('Right now we only know how to handle .jsonl evaluation datasets')

    eval_prefix = args['eval_prefix']

    if args['predictions'] is None:
        # load pretrained or finetuned transformer model
        print(f'loading pre-trained model: {args["model_id"]}')

        # we have to load fine-tuned models in a different way because of pytorch-lightning
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
        args['model'].eval()

        if torch.cuda.is_available():
            args['model'].to('cuda')

        # summarize MDS / summarization dataset with model
        preds_output = open(f'{eval_prefix}eval_predicted_summaries.out', 'w', buffering=1)
        gold_output = open(f'{eval_prefix}eval_gold_summaries.out', 'w', buffering=1)
        metadata_output = open(f'{eval_prefix}decoding_metadata.jsonl', 'w', buffering=1)

        summaries = []
        # get summary for each cluster
        # note here we have a macro-batch size of one cluster by definition
        for cluster in tqdm.tqdm(dataset):
            # shuffle articles before selecting topk to use in ensemble
            articles = [article_to_text(a) for a in cluster['articles']]
            np.random.shuffle(articles)
            articles = articles[:args['max_articles_in_cluster']]

            if args['min_input_char_length'] is not None:
                articles_ = [a for a in articles if len(a) >= args['min_input_char_length']]
                if len(articles_) == 0:
                    articles_ = [articles[0]]
                articles = articles_

            gold_summary = cluster['summary'].strip()

            predictions, sorted_hyps = summarize_articles(articles, args, gold_summary=gold_summary)
            # sorted_hyps -- (token_idxs, score, metadata)
            # they're in sorted order according to ensemble score, so first one is the best
            # we will have one list of timestamp metadata for each cluster input
            length_penalty = args['length_penalty']
            component_scores = []
            for input_idx, state_metadata in enumerate(sorted_hyps[0][2]):
                timestep_scores = np.array([o['score'] for o in state_metadata])
                global_score = np.sum(timestep_scores) / len(timestep_scores) ** length_penalty
                component_scores.append(global_score)

            component_scores = np.array(component_scores)
            for idx in np.argsort(component_scores)[::-1]:
                print(f'ARTICLE: {articles[idx][:1500]}')
                print(f'Input {idx} score: {component_scores[idx]}')
                print()

            print(f'Ensemble score: {sorted_hyps[0][1]}')
            print(f'Gold: {cluster["summary"]}')
            print(f'Predicted: {predictions[0]}')
            print()

            # TODO: sometimes we hit -inf during forced decoding, debug this
            # TODO: if big disparity between article scores, do something, store an input divergence score
            # Note: reverse max / min because scores are logprobs
            if component_scores.max() / sorted(component_scores)[-2] <= .65:
                import ipdb; ipdb.set_trace()

            predicted_summary = predictions[0]
            summaries.append((predicted_summary, gold_summary))
            preds_output.write(f'{predicted_summary}\n')
            gold_output.write(f'{gold_summary}\n')

            sorted_hyps_ = []
            for tok_idxs, score, tok_scores in sorted_hyps:
                tok_idxs = [int(idx) for idx in tok_idxs.cpu().numpy()]
                sorted_hyps_.append((tok_idxs, score, tok_scores))
            sorted_hyps = sorted_hyps_

            metadata_output.write(
                json.dumps(
                    {
                       'cluster': cluster,
                       'predictions': predictions,
                       'inputs_used': articles,
                       'component_scores': list(component_scores),
                       'decoding_metadata': sorted_hyps
                    })
                + '\n')

        preds_output.close()
        gold_output.close()

        # Evaluation
        hyps, refs = zip(*summaries)
    else:
        # Evaluate on user-supplied predictions
        logger.info(f'Evaluating predictions in {args["predictions"]} '
                    f'against gold summaries in {args["evaluation_dataset"]}')
        hyps = [l.strip() for l in open(args['predictions'])]
        # Note this is only single-reference currently
        refs = [json.loads(c)['summary'].strip() for c in open(args['evaluation_dataset'])]
        assert len(hyps) == len(refs)

        # TODO: working -- issue with multi vs single ref setups
        #  - Lebanoff eval requires tokenized predictions -- see what we can do to consolidate evals

    # Ghalandari et al 2020 evaluation
    # TODO: print evaluation results to file
    results, rouge_types = evaluate_rouge(hyps, refs)
    print_mean(results, rouge_types)

    # End evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluation-dataset',
        type=str,
        required=True,
        help='filepath of evaluation data'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=False,
        default=None,
        help='if supplied, evaluation will be done on this output, and new predictions will not be generated'
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
        '--length-penalty',
        type=float,
        required=False,
        default=2.,
        help='length penalty to use when computing final hypothesis scores'
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
        default=5,
        help='take K articles from each cluster to use in the ensemble'
    )
    parser.add_argument(
        '--rows-to-eval',
        type=int,
        required=False,
        default=None,
        help='if provided, truncate eval dataset to this many rows'
    )
    parser.add_argument(
        '--eval-prefix',
        type=str,
        required=False,
        default='',
        help='If provided, prefix of output files'
    )
    parser.add_argument(
        '--force-decode-gold',
        required=False,
        action='store_true',
        help='if this flag is true, we force generation of the gold summary for each cluster'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main(vars(parse_args()))
