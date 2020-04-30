"""
Test stepwise decoding

Then test trivial ensembles (same instance input multiple times)

Then test news summarization ensembles (same instance input multiple times)

"""

import os
from pathlib import Path

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
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

        decoding_hyperparams = {
            'max_length': 75,
            'num_beams': 2
        }

        test_news_article = 'New Zealand says it has stopped community transmission of Covid-19, ' \
                            'effectively eliminating the virus. With new cases in single figures for several days - one on Sunday ' \
                            '- Prime Minister Jacinda Ardern said the virus was "currently" eliminated. But officials have warned ' \
                            'against complacency, saying it does not mean a total end to new coronavirus cases. ' \
                            'The news comes hours before New Zealand is set to move out of its toughest level of social restrictions. ' \
                            'From Tuesday, some non-essential business, healthcare and education activity will be able to resume. ' \
                            'Most people will still be required to remain at home at all times and avoid all social interactions.'

        # set up state
        decoder_state = decoding_utils.get_initial_decoding_state(
            text=test_news_article,
            model=model,
            tokenizer=tokenizer,
            decoding_hyperparams=decoding_hyperparams
        )

        # TODO: move out of test one interfaces are clear
        #  still don't know whether to use BeamHypotheses or not
        # generated hypotheses
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

        # then we wish to step through decoding
        # TODO: note beam logic needs to be outside of step function

        # PLACEHOLDER -- run beam_search_step function
        # ok now we are ready to start stepping
        # step and decode with tokenizer at each step to visualize decoding progress
        for step_idx in range(decoding_hyperparams['max_length']):
            print(f'STEP: {step_idx}')
            print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                   decoder_state['input_ids']])
            decoder_state = decoding_utils.beam_search_step(decoder_state)
            print()
            import ipdb; ipdb.set_trace()

    def test_obtaining_timestep_scores(self):
        """
        Test that we can get the scores out of a model in order to do things with them before deciding upon
         a discrete representation of this timestep and proceeding to the next one.
        """
        pass


if __name__ == '__main__':
    unittest.main()
