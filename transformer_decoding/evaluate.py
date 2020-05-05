import argparse
import json
from pathlib import Path
import tqdm

from transformers import (modeling_utils,
                          BartTokenizer,
                          BartForConditionalGeneration,
                          BartConfig)

import transformer_decoding.log as log

logger = log.create_logger(__name__)


class BartSummarizerConfig:
    def __init__(self, args):
        """
        currently we use the model `bart-large-cnn`
        """
        self.model = BartForConditionalGeneration.from_pretrained('model_name')
        self.tokenizer = BartTokenizer.from_pretrained(args['model_name'])


# TODO: WORKING: in general we want to be able to ensemble both models _and_
#  inputs  -- we should support arbitrary combinations of these, without too much
#  cruft from configuration
def summarize(args):
    model = args['model']
    tokenizer = args['tokenizer']
    decoding_hyperparams = {
        'max_length': args['max_length'],
        'num_beams': args['num_beams']
    }
    dataset = [json.loads(l) for l in open(args['evaluation_dataset'])]

    summaries = []
    # get summary for each cluster
    # note here we have a macro-batch size of one cluster by definition
    for cluster in tqdm.tqdm(dataset):
        articles = [article_to_text(a) for a in cluster['articles'][:args['max_articles']]]

        component_states = [get_start_state(a, model, tokenizer, decoding_hyperparams)
                            for a in articles]
        ensemble_state = get_start_state(articles[0], model, tokenizer, decoding_hyperparams)

        component_states, ensemble_state = \
            decoding_utils.generate(component_states, decoding_hyperparams['max_length'],
                                    ensemble_state=ensemble_state)

        # assert len(ensemble_state['input_ids']) == 1, 'We currently have batch size=1 (we decode one cluster at a time)'
        print(f'input_ids shape: {ensemble_state["input_ids"].shape}')
        print(f'Reference Summary:\n{cluster["summary"]}')
        predictions = [tokenizer.decode(input_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                       for input_ids in ensemble_state['input_ids']]
        print(f'Predictions: \n{predictions}')
        print()

        # NOTE: hack to just take the first one right now
        summaries.append((predictions[0], cluster['summary']))
    return summaries


class Summarizer:

    def __init__(self, config):
        model = config['model']
        tokenizer = config['tokenizer']

    # TODO: factor out score computation vs reduction




def main(args):

    # load pretrained or finetuned transformer model

    # summarize MDS / summarization dataset with model

    # print and write out evaluation results


    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resource-path',
        type=str,
        required=True,
        help='path to resources'
    )
    parser.add_argument(
        '--evaluation-dataset',
        type=str,
        required=True,
        help='filepath of evaluation data'
    )
    parser.add_argument(
        '--spotter',
        type=str,
        default=None,
        required=False,
        help='name of the spotter to use in the linking pipeline'
    )
    parser.add_argument(
        '--linker',
        type=str,
        default='FastTextEntityLinker',
        required=False,
        help='name of the linker class to use'
    )
    parser.add_argument(
        '--strict-boundary',
        dest='strict_boundary',
        action='store_true',
        help='If used, entities match only if both start and end are '
             'the same'
    )
    parser.add_argument(
        '--wp-to-wd-file',
        type=str,
        default=None,
        required=False,
        help='locatiion of the map from wikiname to wikdata id for '
             'ConceptsEndpointLinker'
    )
    parser.add_argument(
        '--concepts-endpoint-url',
        type=str,
        default=None,
        required=False,
        help='the location where the concepts endpoint is running, used '
             'by ConceptsEndpointLinker'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(vars(parse_args()))
