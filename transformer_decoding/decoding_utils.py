from collections import OrderedDict

import torch
from torch.nn import functional as F

from transformers import modeling_utils

from transformer_decoding import decoding_utils, log


logger = log.create_logger(__name__)


def generate(component_states, timesteps, ensemble_state=None):
    """
    Run generation for a number of timesteps
    """
    if type(component_states) is not list:
        component_states = [component_states]

    if ensemble_state is None:
        assert len(component_states) == 1
        for step_idx in range(timesteps):
            component_states[0] = decoding_utils.beam_search_step(component_states[0])
    else:
        for step_idx in range(timesteps):
            component_states, ensemble_state = \
                decoding_utils.ensembled_beam_search_step(component_states, ensemble_state)

    return component_states, ensemble_state


def initialize_generation(
        model,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
):
    # We cannot generate if the model does not have a LM head
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    temperature = temperature if temperature is not None else model.config.temperature
    top_k = top_k if top_k is not None else model.config.top_k
    top_p = top_p if top_p is not None else model.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(model.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    vocab_size = model.config.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
                decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()

        encoder_outputs = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    # Chris: note this is important distinction between decoder-only and
    # encoder-decoder architectures, for encoder-decoder models, decoder state
    # is initialized with decoder BOS token, for decoder-only models, it's the whole
    # input_ids as passed to this function
    # Note: in the current formulation this precludes prefix-completion usecases by definition
    if model.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1

        assert (
                batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    # Chris: return the outputs needed for model._generate_beam_search
    return OrderedDict([
        ('model', model),
        ('input_ids', input_ids),
        ('cur_len', cur_len),
        ('max_length', max_length),
        ('min_length', min_length),
        ('do_sample', do_sample),
        ('early_stopping', early_stopping),
        ('temperature', temperature),
        ('top_k', top_k),
        ('top_p', top_p),
        ('repetition_penalty', repetition_penalty),
        ('no_repeat_ngram_size', no_repeat_ngram_size),
        ('bad_words_ids', bad_words_ids),
        ('bos_token_id', bos_token_id),
        ('pad_token_id', pad_token_id),
        ('decoder_start_token_id', decoder_start_token_id),
        ('eos_token_id', eos_token_id),
        ('batch_size', effective_batch_size),
        ('num_return_sequences', num_return_sequences),
        ('length_penalty', length_penalty),
        ('num_beams', num_beams),
        ('vocab_size', vocab_size),
        ('encoder_outputs', encoder_outputs),
        ('attention_mask', attention_mask)
    ])


# TODO: remember critical assumption that all models use the same output space, we need to use this during
#  ensembling -- remember token / segment level ensembling ideas (token probability is mean of constituent subwords)
# TODO: does model instance hold any state while decoding? I.e. is model's self.* holding any state while we are
#  inside the decoding loop?
# TODO: remove decoding length arg

# WORKING: wrap tokenizer and model together so that we can pass through (text, tokenizer, model)
def get_initial_decoding_state(text, model, tokenizer, decoding_hyperparams):
    """
    Get the state needed to start decoding from an instance
    """
    # convert text to tensor
    inputs = tokenizer.batch_encode_plus(
        [text],
        max_length=decoding_hyperparams['max_length'],
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']

    return initialize_generation(
        model, input_ids,
        **decoding_hyperparams
    )


def outputs_from_state(state):
    """
    Run forward pass using a state, note this only works for states with a 'model' attribute
    """
    model_inputs = state['model'].prepare_inputs_for_generation(
        state['input_ids'],
        past=state['past'],
        attention_mask=state['attention_mask'])
    outputs = state['model'](**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
    return outputs


def logits_from_output(state):
    """
    In the context of ensemble decoding, decoding parameters may be applied twice
     - once on individual states
     - once on the entire ensemble
    As currently implemented, some decoding heuristics are applied to the logits,
     some are applied to the scores (logits after softmax).
    """
    pass


def apply_heuristics_to_logits(state):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if state['repetition_penalty'] != 1.0:
        state['model'].enforce_repetition_penalty_(
            state['next_token_logits'],
            state['batch_size'],
            state['num_beams'],
            state['input_ids'],
            state['repetition_penalty']
        )

    if state['temperature'] != 1.0:
        state['next_token_logits'] = state['next_token_logits'] / state['temperature']

    return state


@torch.no_grad()
def ensembled_beam_search_step(component_states, ensemble_state):
    """
    Decoding hyperparams live in ensemble_state
    """

    for state in component_states:
        state['outputs'] = outputs_from_state(state)
        state['next_token_logits'] = state['outputs'][0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        state = apply_heuristics_to_logits(state)
        # apply softmax to logits
        state['scores'] = F.log_softmax(state['next_token_logits'], dim=-1)  # (batch_size * num_beams, vocab_size)

        if state['model'].config.is_encoder_decoder and ensemble_state['do_sample'] is False:
            # TODO (PVP) still a bit hacky here - there might be a better solution
            state['scores'] = state['model'].prepare_scores_for_generation(
                state['scores'],
                cur_len=state['cur_len'],
                max_length=state['max_length'])

        # set state's eos token prob to zero if min_length is not reached
        if ensemble_state['eos_token_id'] is not None and ensemble_state['cur_len'] < ensemble_state['min_length']:
            state['scores'][:, state['eos_token_id']] = -float("inf")

        if ensemble_state['no_repeat_ngram_size'] > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = ensemble_state['batch_size'] * ensemble_state['num_beams']
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = modeling_utils.calc_banned_ngram_tokens(
                ensemble_state['input_ids'],
                num_batch_hypotheses,
                ensemble_state['no_repeat_ngram_size'],
                ensemble_state['cur_len']
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                state['scores'][i, banned_tokens] = -float("inf")

        if ensemble_state['bad_words_ids'] is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = modeling_utils.calc_banned_bad_words_ids(
                ensemble_state['input_ids'],
                ensemble_state['bad_words_ids']
            )

            for i, banned_tokens in enumerate(banned_tokens):
                state['scores'][i, banned_tokens] = -float("inf")

        assert state['scores'].shape == (
            ensemble_state['batch_size'] * ensemble_state['num_beams'], ensemble_state['vocab_size']), "Shapes of scores: {} != {}".format(
            state['scores'].shape, (ensemble_state['batch_size'] * ensemble_state['num_beams'], ensemble_state['vocab_size'])
        )

        # TODO: put this side-effect somewhere reasonable
        # if model has past, then set the past variable to speed up decoding
        if state['model']._do_output_past(state['outputs']):
            state['past'] = state['outputs'][1]

    # CHRIS: WORKING HERE
    # - assume (and enforce) that component state input_ids are always the same as ensemble state's input_ids
    # - note potential complexity in re-ordering due to individual model's different encoder_outputs

    # TODO: now we have attached scores to every individual model's state, let's proceed to update the ensemble state
    # TODO: note individual models should not need to care about beam search outside of the necessary reordering of inputs

    # Chris: ok, now we have the scores from this (model, text) pair, let's return them and ensemble before
    #  continuing.
    # Chris: let's create a wrapper that holds pairs of model, text
    # Chris: let's create a new type of hypothesis which stores additional metadata in the beam
    # Chris: same structure as beam, but stores arbitrary meta-data in each cell -- WORKING: what is the "timestamp metatdata?"

    # TODO: now call the reduce function over all state['scores'], this will give us ensemble_state['scores']
    # REDUCE SCORES AND SET ONTO ENSEMBLE STATE
    # from old implementation:
    #ensembled_log_probs = \
    #    torch.mean(
    #        timestep_weights[:, None] * torch.squeeze(torch.stack(cohort_log_probs)),
    #        dim=0
    #    )[None, :]

    # TODO: just simple mean of logprobs as first try, later more sophisticated weighting
    ensemble_state['scores'] = torch.mean(torch.stack([s['scores'] for s in component_states]), dim=0)

    # BEGIN: ways of selecting next token from scores
    if ensemble_state['do_sample']:
        # TEMP
        pass
        # END TEMP

        _scores = scores + state['beam_scores'][:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # Top-p/top-k filtering
        # Chris: note hard-coded `min_tokens_to_keep`
        _scores = modeling_utils.top_k_top_p_filtering(
            _scores, top_k=state['top_k'], top_p=state['top_p'], min_tokens_to_keep=2
        )  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together to sample from all beam_idxs
        _scores = _scores.contiguous().view(
            state['batch_size'], state['num_beams'] * state['vocab_size']
        )  # (batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
        probs = F.softmax(_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=2 * state['num_beams'])  # (batch_size, num_beams * 2)
        # Compute next scores
        next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
        # sort the sampled vector to make sure that the first num_beams samples are the best
        next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

    else:
        next_scores = ensemble_state['scores'] + ensemble_state['beam_scores'][:, None].expand_as(ensemble_state['scores'])  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypotheses across beams)
        next_scores = next_scores.view(
            ensemble_state['batch_size'], ensemble_state['num_beams'] * ensemble_state['vocab_size']
        )  # (batch_size, num_beams * vocab_size)

        # import ipdb; ipdb.set_trace()

        # Chris: there is a |vocab| * beam_idx offset
        next_scores, next_tokens = \
            torch.topk(
                next_scores,
                2 * ensemble_state['num_beams'],
                dim=1,
                largest=True,
                sorted=True
            )

    assert next_scores.size() == next_tokens.size() == (ensemble_state['batch_size'], 2 * ensemble_state['num_beams'])
    # NEXT TOKEN CANDIDATES HAVE BEEN SELECTED

    # BEGIN: UPDATING SEARCH STATE
    # next batch beam content
    next_batch_beam = []

    # for each sentence (note if we are doing one multi-doc summary, batch_size is 1 for sure)
    for batch_idx in range(ensemble_state['batch_size']):

        # if we are done with this sentence
        if ensemble_state['done'][batch_idx]:
            assert (
                    len(ensemble_state['generated_hyps'][batch_idx]) >= ensemble_state['num_beams']
            ), "Batch can only be done if at least {} beams have been generated".format(state['num_beams'])
            assert (
                    ensemble_state['eos_token_id'] is not None and ensemble_state['pad_token_id'] is not None
            ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
            next_batch_beam.extend([(0, ensemble_state['pad_token_id'], 0)] * ensemble_state['num_beams'])  # pad the batch
            continue

        # next sentence beam content
        next_sent_beam = []

        # next tokens for this sentence from each beam
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
        ):
            # get beam and token IDs (undo beam offset)
            beam_id = beam_token_id // ensemble_state['vocab_size']
            token_id = beam_token_id % ensemble_state['vocab_size']

            effective_beam_id = batch_idx * ensemble_state['num_beams'] + beam_id
            # import ipdb; ipdb.set_trace()
            # add to generated hypotheses if end of sentence or last iteration
            if (ensemble_state['eos_token_id'] is not None) and (token_id.item() == ensemble_state['eos_token_id']):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= ensemble_state['num_beams']
                if is_beam_token_worse_than_top_num_beams:
                    continue
                # update beam hypotheses obj with finished hypothesis and score
                ensemble_state['generated_hyps'][batch_idx].add(
                    ensemble_state['input_ids'][effective_beam_id].clone(), beam_token_score.item(),
                )
            else:
                # add next predicted token if it is not eos_token
                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

            # the beam for next step is now full
            if len(next_sent_beam) == ensemble_state['num_beams']:
                break

        # import ipdb; ipdb.set_trace()

        # Check if we're done so that we can save a pad step if all(done)
        ensemble_state['done'][batch_idx] = ensemble_state['done'][batch_idx] or ensemble_state['generated_hyps'][batch_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len=ensemble_state['cur_len']
        )

        # update next beam content
        assert len(next_sent_beam) == ensemble_state['num_beams'], "Beam should always be full after loop above"
        next_batch_beam.extend(next_sent_beam)
        assert len(next_batch_beam) == ensemble_state['num_beams'] * (batch_idx + 1)

    # stop if are done with every sentence
    if all(ensemble_state['done']):
        return component_states, ensemble_state

    # sanity check / prepare next timestep
    assert len(next_batch_beam) == ensemble_state['batch_size'] * ensemble_state['num_beams']

    # Note we shouldn't need to deal with beam scores on the component_states
    # Chris: the score of each item is this timestep's score + previous beam score
    ensemble_state['beam_scores'] = ensemble_state['beam_scores'].new([x[0] for x in next_batch_beam])

    # re-order batch
    beam_tokens = ensemble_state['input_ids'].new([x[1] for x in next_batch_beam])
    beam_idx = ensemble_state['input_ids'].new([x[2] for x in next_batch_beam])

    # TODO: possible bug land here
    # TODO: WORKING: set input_ids, cur_len, past on all component states and on ensemble_state
    for state in component_states:
        state['input_ids'] = ensemble_state['input_ids'][beam_idx, :]
        state['input_ids'] = torch.cat([ensemble_state['input_ids'], beam_tokens.unsqueeze(1)], dim=-1)

    ensemble_state['input_ids'] = ensemble_state['input_ids'][beam_idx, :]
    ensemble_state['input_ids'] = torch.cat([ensemble_state['input_ids'], beam_tokens.unsqueeze(1)], dim=-1)

    # import ipdb; ipdb.set_trace()

    # re-order internal states
    # Note ensemble_state has no "past", this is only on component_states
    # TODO: Note in case batch size is 1 (beam can be larger), all 'past' should be identical, so this reordering shouldn't matter
    # TODO: confirm this as it could lead to very weird bugs
    for state in component_states:
        state['past'] = state['model']._reorder_cache(state['past'], beam_idx)

    # extend attention_mask for new generated input if only decoder
    # Chris: commented until we need a decoder-only model
    #if state['model'].config.is_encoder_decoder is False:
    #    state['attention_mask'] = torch.cat(
    #        [
    #            state['attention_mask'],
    #            state['attention_mask'].new_ones((state['attention_mask'].shape[0], 1))
    #        ],
    #        dim=-1
    #    )

    # update current length
    for state in component_states:
        state['cur_len'] = state['cur_len'] + 1

    ensemble_state['cur_len'] = ensemble_state['cur_len'] + 1

    return component_states, ensemble_state

@torch.no_grad()
def beam_search_step(state):
    if state.get('outputs', None) is None:
        outputs = outputs_from_state(state)

    next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

    # if model has past, then set the past variable to speed up decoding
    if state['model']._do_output_past(outputs):
        state['past'] = outputs[1]

    import ipdb; ipdb.set_trace()

    # TODO: Chris working: some heuristics are applied in-place to logits, others to scores
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if state['repetition_penalty'] != 1.0:
        state['model'].enforce_repetition_penalty_(
            next_token_logits,
            state['batch_size'],
            state['num_beams'],
            state['input_ids'],
            state['repetition_penalty']
        )

    if state['temperature'] != 1.0:
        next_token_logits = next_token_logits / state['temperature']

    scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
    if state['model'].config.is_encoder_decoder and state['do_sample'] is False:
        # TODO (PVP) still a bit hacky here - there might be a better solution
        scores = state['model'].prepare_scores_for_generation(
            scores,
            cur_len=state['cur_len'],
            max_length=state['max_length'])

    # set eos token prob to zero if min_length is not reached
    if state['eos_token_id'] is not None and state['cur_len'] < state['min_length']:
        scores[:, state['eos_token_id']] = -float("inf")

    if state['no_repeat_ngram_size'] > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = state['batch_size'] * state['num_beams']
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = modeling_utils.calc_banned_ngram_tokens(
            state['input_ids'],
            num_batch_hypotheses,
            state['no_repeat_ngram_size'],
            state['cur_len']
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if state['bad_words_ids'] is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = modeling_utils.calc_banned_bad_words_ids(
            state['input_ids'],
            state['bad_words_ids']
        )

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    assert scores.shape == (
        state['batch_size'] * state['num_beams'], state['vocab_size']), "Shapes of scores: {} != {}".format(
        scores.shape, (state['batch_size'] * state['num_beams'], state['vocab_size'])
    )

    # Chris: ok, now we have the scores from this (model, text) pair, let's return them and ensemble before
    #  continuing.
    # Chris: let's create a wrapper that holds pairs of model, text
    # Chris: let's create a new type of hypothesis which stores additional metadata in the beam
    # Chris: same structure as beam, but stores arbitrary meta-data in each cell -- WORKING: what is the "timestamp metatdata?"

    # BEGIN: ways of selecting next token from scores
    if state['do_sample']:
        _scores = scores + state['beam_scores'][:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # Top-p/top-k filtering
        # Chris: note hard-coded `min_tokens_to_keep`
        _scores = modeling_utils.top_k_top_p_filtering(
            _scores, top_k=state['top_k'], top_p=state['top_p'], min_tokens_to_keep=2
        )  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together to sample from all beam_idxs
        _scores = _scores.contiguous().view(
            state['batch_size'], state['num_beams'] * state['vocab_size']
        )  # (batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
        probs = F.softmax(_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=2 * state['num_beams'])  # (batch_size, num_beams * 2)
        # Compute next scores
        next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
        # sort the sampled vector to make sure that the first num_beams samples are the best
        next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

    else:
        next_scores = scores + state['beam_scores'][:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypotheses across beams)
        next_scores = next_scores.view(
            state['batch_size'], state['num_beams'] * state['vocab_size']
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = \
            torch.topk(
                next_scores,
                2 * state['num_beams'],
                dim=1,
                largest=True,
                sorted=True
            )

    assert next_scores.size() == next_tokens.size() == (state['batch_size'], 2 * state['num_beams'])
    # NEXT TOKEN CANDIDATES HAVE BEEN SELECTED

    # BEGIN: UPDATING SEARCH STATE
    # next batch beam content
    next_batch_beam = []

    # for each sentence
    for batch_idx in range(state['batch_size']):

        # if we are done with this sentence
        if state['done'][batch_idx]:
            assert (
                    len(state['generated_hyps'][batch_idx]) >= state['num_beams']
            ), "Batch can only be done if at least {} beams have been generated".format(state['num_beams'])
            assert (
                    state['eos_token_id'] is not None and state['pad_token_id'] is not None
            ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
            next_batch_beam.extend([(0, state['pad_token_id'], 0)] * state['num_beams'])  # pad the batch
            continue

        # next sentence beam content
        next_sent_beam = []

        # next tokens for this sentence from each beam
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
        ):
            # get beam and token IDs
            beam_id = beam_token_id // state['vocab_size']
            token_id = beam_token_id % state['vocab_size']

            effective_beam_id = batch_idx * state['num_beams'] + beam_id
            # add to generated hypotheses if end of sentence or last iteration
            if (state['eos_token_id'] is not None) and (token_id.item() == state['eos_token_id']):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= state['num_beams']
                if is_beam_token_worse_than_top_num_beams:
                    continue
                # update beam hypotheses obj with finished hypothesis and score
                state['generated_hyps'][batch_idx].add(
                    state['input_ids'][effective_beam_id].clone(), beam_token_score.item(),
                )
            else:
                # add next predicted token if it is not eos_token
                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

            # the beam for next step is now full
            if len(next_sent_beam) == state['num_beams']:
                break

        # Check if we're done so that we can save a pad step if all(done)
        state['done'][batch_idx] = state['done'][batch_idx] or state['generated_hyps'][batch_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len=state['cur_len']
        )

        # update next beam content
        assert len(next_sent_beam) == state['num_beams'], "Beam should always be full after loop above"
        next_batch_beam.extend(next_sent_beam)
        assert len(next_batch_beam) == state['num_beams'] * (batch_idx + 1)

    # stop if are done with every sentence
    if all(state['done']):
        return state

    # sanity check / prepare next timestep
    assert len(next_batch_beam) == state['batch_size'] * state['num_beams']
    state['beam_scores'] = state['beam_scores'].new([x[0] for x in next_batch_beam])

    # re-order batch
    beam_tokens = state['input_ids'].new([x[1] for x in next_batch_beam])
    beam_idx = state['input_ids'].new([x[2] for x in next_batch_beam])

    state['input_ids'] = state['input_ids'][beam_idx, :]
    state['input_ids'] = torch.cat([state['input_ids'], beam_tokens.unsqueeze(1)], dim=-1)
    # re-order internal states
    if state['past'] is not None:
        state['past'] = state['model']._reorder_cache(state['past'], beam_idx)

    # extend attention_mask for new generated input if only decoder
    if state['model'].config.is_encoder_decoder is False:
        state['attention_mask'] = torch.cat(
            [
                state['attention_mask'],
                state['attention_mask'].new_ones((state['attention_mask'].shape[0], 1))
            ],
            dim=-1
        )

    # update current length
    state['cur_len'] = state['cur_len'] + 1
    return state


# this is def step() for model._generate_no_beam_search
@torch.no_grad()
def greedy_step(state):
    model_inputs = state['model'].prepare_inputs_for_generation(
        state['input_ids'],
        past=state['past'],
        attention_mask=state['attention_mask']
    )

    outputs = state['model'](**model_inputs)
    next_token_logits = outputs[0][:, -1, :]

    # if model has past, then set the past variable to speed up decoding
    if state['model']._do_output_past(outputs):
        state['past'] = outputs[1]

    # now update next_token_logits using various heuristics

    # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
    if state['repetition_penalty'] != 1.0:
        # Chris: note in-place modification side-effect
        state['model'].enforce_repetition_penalty_(
            next_token_logits,
            state['batch_size'], 1, state['input_ids'], state['repetition_penalty'])

    if state['no_repeat_ngram_size'] > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345

        banned_tokens = modeling_utils.calc_banned_ngram_tokens(
            state['input_ids'],
            state['batch_size'],
            state['no_repeat_ngram_size'],
            state['cur_len'])
        for batch_idx in range(state['batch_size']):
            next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

    if state['bad_words_ids'] is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = modeling_utils.calc_banned_bad_words_ids(state['input_ids'], state['bad_words_ids'])

        for batch_idx in range(state['batch_size']):
            next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

    # Chris: WORKING: note any next token logic must live outside of the step function
    # Chris: put this into codebase first before proceeding with TDD

    # set eos token prob to zero if min_length is not reached
    if state['eos_token_id'] is not None and state['cur_len'] < state['min_length']:
        next_token_logits[:, state['eos_token_id']] = -float("inf")

    if state['do_sample']:
        # Temperature (higher temperature => more likely to sample low probability tokens)
        if state['temperature'] != 1.0:
            next_token_logits = next_token_logits / state['temperature']
        # Top-p/top-k filtering
        next_token_logits = \
            modeling_utils.top_k_top_p_filtering(
                next_token_logits,
                top_k=state['top_k'],
                top_p=state['top_p']
            )
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        # Chris: TODO: note for ensembling all next token logic
        #  needs to move outside of this function
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        # Greedy decoding
        # Chris: TODO: note for ensembling all next token logic needs to move outside of this function
        next_token = torch.argmax(next_token_logits, dim=-1)

    # Chris: TODO: update unfinished_sents in state
    # update generations and finished sentences
    if state['eos_token_id'] is not None:
        # pad finished sentences if eos_token_id exist
        tokens_to_add = next_token * state['unfinished_sents'] + (state['pad_token_id']) * (
                1 - state['unfinished_sents'])
    else:
        tokens_to_add = next_token

    # Chris: concat whatever was generated to input ids
    # Chris: TODO: this must happen outside of individual model's step functions
    state['input_ids'] = torch.cat([state['input_ids'], tokens_to_add.unsqueeze(-1)], dim=-1)

    if state['eos_token_id'] is not None:
        eos_in_sents = tokens_to_add == state['eos_token_id']
        # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        is_sents_unfinished_and_token_to_add_is_eos = state['unfinished_sents'].mul(eos_in_sents.long()).bool()
        state['sent_lengths'].masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, state['cur_len'] + 1)
        # unfinished_sents is set to zero if eos in sentence
        state['unfinished_sents'].mul_((~eos_in_sents).long())

    # stop when there is a </s> in each sentence, or if we exceed the maximal length
    if state['unfinished_sents'].max() == 0:
        return state

    # extend attention_mask for new generated input if only decoder
    if state['model'].config.is_encoder_decoder is False:
        state['attention_mask'] = torch.cat(
            [state['attention_mask'],
             state['attention_mask'].new_ones((state['attention_mask'].shape[0], 1))],
            dim=-1
        )

    state['cur_len'] = state['cur_len'] + 1

    return state



