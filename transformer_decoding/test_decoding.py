"""
Test stepwise decoding

Then test trivial ensembles (same instance input multiple times)

Then test news summarization ensembles (same instance input multiple times)

"""

import os
from pathlib import Path
import copy

from collections import OrderedDict

import torch

import transformers
from transformers import (modeling_utils,
                          BartTokenizer,
                          BartForConditionalGeneration,
                          BartConfig)

import unittest

from transformer_decoding import decoding_utils


#path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
#resources = Path(
#    os.environ.get('RESOURCES', path_to_file / '../resources/test')
#)


def get_start_state(text, model, tokenizer, decoding_hyperparams):
    # set up state
    decoder_state = decoding_utils.get_initial_decoding_state(
        text=text,
        model=model,
        tokenizer=tokenizer,
        decoding_hyperparams=decoding_hyperparams
    )

    # TODO: move out of test one interfaces are clear
    #  still don't know whether to use BeamHypotheses or not
    #  generated hypotheses -- this may move to
    #  `get_initial_decoding_state`
    decoder_state['generated_hyps'] = [
        modeling_utils.BeamHypotheses(
            decoder_state['num_beams'],
            decoder_state['max_length'],
            decoder_state['length_penalty'],
            early_stopping=decoder_state['early_stopping'])
        for _ in range(decoder_state['batch_size'])
    ]

    # scores for each sentence in the beam
    decoder_state['beam_scores'] = \
        torch.zeros((decoder_state['batch_size'], decoder_state['num_beams']),
                    dtype=torch.float,
                    device=decoder_state['input_ids'].device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered
    #  to avoid sampling the exact same tokens three times
    if decoder_state['do_sample'] is False:
        decoder_state['beam_scores'][:, 1:] = -1e9
    decoder_state['beam_scores'] = decoder_state['beam_scores'].view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    decoder_state['past'] = decoder_state[
        'encoder_outputs']  # defined for encoder-decoder models, None for decoder-only models

    # done sentences
    decoder_state['done'] = [False for _ in range(decoder_state['batch_size'])]

    return decoder_state


# Set up transformer model with LM head then assert things
# TODO: which transformer models have encoder-->decoder
class TestTransformerDecoding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # summarization
        # generate yes beam search
        # Note for BART summarization in transformers repo, beam search performs much better
        #  than no beam search, but even their beam search with num_beams=1 is better, implying that something
        #  is broken in the _generate_no_beam_search function

        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        cls.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        cls.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

        cls.decoding_hyperparams = {
            'max_length': 75,
            'num_beams': 2
        }

        cls.test_news_article_1 = 'New Zealand says it has stopped community transmission of Covid-19, ' \
                            'effectively eliminating the virus. With new cases in single figures for several days - one on Sunday ' \
                            '- Prime Minister Jacinda Ardern said the virus was "currently" eliminated. But officials have warned ' \
                            'against complacency, saying it does not mean a total end to new coronavirus cases. ' \
                            'The news comes hours before New Zealand is set to move out of its toughest level of social restrictions. ' \
                            'From Tuesday, some non-essential business, healthcare and education activity will be able to resume. ' \
                            'Most people will still be required to remain at home at all times and avoid all social interactions.'

        cls.test_news_article_2 = \
            'But officials have warned against complacency, saying it does not mean a total end to new HIV cases. ' \
            'Most people will still be required to remain at home at all times and avoid all social interactions.' \
            'Germany says it has stopped community transmission of HIV, ' \
            'effectively eliminating the virus. With new cases in single figures for several days - one on Sunday ' \
            '- Prime Minister Angela Merkle said the virus was "currently" eliminated. ' \
            'From Tuesday, some non-essential business, healthcare and education activity will be able to resume. ' \
            'The news comes hours before Germany is set to move out of its toughest level of social restrictions. '

    def test_obtaining_timestep_scores(self):
        """
        Test that we can get the scores out of a model in order to do things with them before deciding upon
         a discrete representation of this timestep and proceeding to the next one.
        """
        # then we wish to step through decoding
        # TODO: note beam logic needs to be outside of step function

        # WORKING: next step -- split scoring and beam consolidation into separate functions

        # for summarization, args on initial state which are input-specific:
        # decoder_state['model']
        # decoder_state['encoder_outputs']

        # decoder_state['past'] will also hold something model-specific(?)
        # Every other arg is a decoding hyperparam

        # as decoding proceeds, input_ids will hold current state
        # TODO: IDEA: pass a list of states, and one additional state to hold their combined outputs
        # TODO: IDEA: new function `def get_ensemble_wrapper_decoding_state`

        decoder_state_1 = get_start_state(
            self.test_news_article_2,
            self.model,
            self.tokenizer,
            self.decoding_hyperparams)

        decoder_state_2 = get_start_state(
            self.test_news_article_2,
            self.model,
            self.tokenizer,
            self.decoding_hyperparams)

        #first_step_outputs = decoding_utils.outputs_from_state(decoder_state)

        # run beam_search_step function
        # ok now we are ready to start stepping
        # step and decode with tokenizer at each step to visualize and understand decoding progress
        # for step_idx in range(decoding_hyperparams['max_length']):
        #    print(f'STEP: {step_idx}')
        #    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
        #           decoder_state['input_ids']])
        #    decoder_state = decoding_utils.beam_search_step(decoder_state)
        #    print()
        #    import ipdb; ipdb.set_trace()

        component_states = [decoder_state_1, decoder_state_2]

        # TODO: assert that it doesn't matter which state we initialize ensemble_state from
        # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
        ensemble_state = OrderedDict()
        for k, v in decoder_state_1.items():
            try:
                ensemble_state[k] = copy.deepcopy(v)
            except:
                print(f'Can\'t copy state item: {k}, type: {type(v)}')
                if len(v):
                    print(f'types of items in v: {[type(c) for c in v]}')

        for step_idx in range(self.decoding_hyperparams['max_length']):
            print(f'STEP: {step_idx}')
            print([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                   ensemble_state['input_ids']])
            component_states, ensemble_state = \
                decoding_utils.ensembled_beam_search_step(component_states, ensemble_state)
            print()
            if step_idx % 10 == 0:
                import ipdb; ipdb.set_trace()

        # TODO: assert that single-member ensemble and two-member identical ensemble give
        #  same results as non-ensembled beam search


if __name__ == '__main__':
    unittest.main()
