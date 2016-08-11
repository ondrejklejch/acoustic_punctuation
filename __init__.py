import logging
import theano

from collections import Counter

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, CompositeRule, RemoveNotFinite)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model

from helpers import create_model, create_multitask_model
from checkpoint import CheckpointNMT, LoadNMT
from sampling import F1Validator, Sampler

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

theano.config.on_unused_input = 'warn'
theano.config.exception_verbosity = 'low'


def main(config, tr_stream, dev_stream, use_bokeh=False):

    logger.info('Building RNN encoder-decoder')
    cost, samples, search_model = create_model(config)
    #cost, samples, search_model = create_multitask_model(config)

    logger.info("Building model")
    cg = ComputationGraph(cost)
    training_model = Model(cost)


    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'], every_n_batches=config['save_freq'])
    ]

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    src_vocab=config['src_vocab'], trg_vocab=config['trg_vocab'],
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on f1
    if config['f1_validation'] is not None:
        logger.info("Building f1 validator")
        extensions.append(
            F1Validator(samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_f1'],
                          every_n_batches=config['f1_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']), eval(config['step_rule'])(), RemoveNotFinite()]),
        on_unused_sources='warn'
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()
