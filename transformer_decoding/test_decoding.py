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
        cls.model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        cls.tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

        cls.decoding_hyperparams = {
            'max_length': 40,
            'num_beams': 3
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
        # for summarization, args on initial state which are input-specific:
        # decoder_state['model']
        # decoder_state['encoder_outputs']
        # decoder_state['past'] will also hold something model-specific(?)
        # Every other arg is a decoding hyperparam

        # as decoding proceeds, input_ids will hold current state
        # IDEA: pass a list of states, and one additional state to hold their combined outputs
        test_articles_1 = [self.test_news_article_1, self.test_news_article_2]
        component_states_1 = [get_start_state(a, self.model, self.tokenizer, self.decoding_hyperparams)
                            for a in test_articles_1]
        ensemble_state_1 = get_start_state(test_articles_1[0], self.model, self.tokenizer, self.decoding_hyperparams)

        # TODO: at the beginning of decoding, the ensemble state doesn't know anything about the component states
        #  - we should try to encode this explicitly by _not_ passing an input to initialize this state
        # TODO: remove past and encoder outputs from ensemble state
        # TODO: remove decoding hyperparams from component_states for sanity

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

        # TODO: assert that it doesn't matter which state we initialize ensemble_state from
        component_states_1, ensemble_state_1 = \
            decoding_utils.generate(component_states_1, self.decoding_hyperparams['max_length'],
                                    ensemble_state=ensemble_state_1)

        # reorder articles and run again
        test_articles_2 = [self.test_news_article_2, self.test_news_article_1]
        component_states_2 = [get_start_state(a, self.model, self.tokenizer, self.decoding_hyperparams)
                              for a in test_articles_2]
        ensemble_state_2 = get_start_state(test_articles_2[0], self.model, self.tokenizer, self.decoding_hyperparams)

        component_states_2, ensemble_state_2 = \
            decoding_utils.generate(component_states_2, self.decoding_hyperparams['max_length'],
                                    ensemble_state=ensemble_state_2)

        for o1_ids, o2_ids in zip(ensemble_state_1['input_ids'], ensemble_state_2['input_ids']):
            o1_text = self.tokenizer.decode(o1_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            o2_text = self.tokenizer.decode(o2_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f'o1_text: {o1_text}')
            print(f'o1_ids: {o1_ids}')
            assert o1_text == o2_text


if __name__ == '__main__':
    unittest.main()
