#!/usr/bin/env python3

# Copyright 2018 David B. Adrian, Mercateo AG (http://www.mercateo.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=========================================================================


try:
    from deso.argcomp import CompletingArgumentParser as ArgumentParser
except ImportError:
    from argparse import ArgumentParser

import json
import logging
import os
import sys

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import eknog.utils as common
import eknog.datasets as datasets
import eknog.models as models
import eknog.tf_helpers as tf_helpers
from eknog.exceptions import ModelInvalid, ModelParametersInvalid
from eknog.trainer import Trainer

common.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def generate_tf_config(args):
    if args.num_gpu == -1:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # config.intra_op_parallelism_threads = 2
    # config.inter_op_parallelism_threads = 2
    return config


def args_debug(func):
    def func_wrapper(args):
        logger.debug("Called < {} > with args: {}".format(func.__name__, args))
        return func(args)

    return func_wrapper


@args_debug
def train(args):
    # Load Dataset in this section
    if args.dataset in datasets.__all__:
        dataset = getattr(datasets, args.dataset)()
    else:
        logger.info(
            "Loading dataset by name with BaseDataset class. This is the normal behavior, but might be unintended.")
        dataset = datasets.BaseDataset(dataset=args.dataset)
        # raise DatasetInvalid

    # Load Model
    if args.model in models.__all__:
        try:
            if args.num_gpu == 1:
                logger.info("Single GPU Mode: Fixing variables to /gpu:0")
                args.model_params["variable_device"] = "/gpu:0"

            model = getattr(models, args.model)(dataset=dataset,
                                                **args.model_params)
        except TypeError as e:
            logger.error("Invalid Keyword/Argument supplied: ", e)
            raise ModelParametersInvalid
    else:
        raise ModelInvalid

    optimizer = "AdamOptimizer"
    # optimizer = "AdagradOptimizer"
    if args.optimizer_params:
        if "optimizer" in args.optimizer_params:
            optimizer = args.optimizer_params["optimizer"]
            args.optimizer_params.pop("optimizer")

        optimizer_settings = args.optimizer_params
    else:
        optimizer_settings = {
            "learning_rate": 0.003
        }

    # Generate a TF-Session Confg (e.g., to activate debug mode)
    tf_config = generate_tf_config(args)

    with tf.Session(config=tf_config) as sess:

        # Wrap session by debug session
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        trainer_config = args.trainer_params
        trainer_config["tf_config"] = tf_config
        trainer_config["optimizer"] = optimizer
        trainer_config["optimizer_settings"] = optimizer_settings
        trainer_config["timeline"] = args.timeline
        trainer_config["num_gpu"] = args.num_gpu
        trainer_config["tag"] = args.tag

        trainer = Trainer(dataset, model, sess, **trainer_config)
        logger.debug(trainer.config)

        # Finally initialize everything else (e.g., optimizer)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Setting up model via its init ops and follows prepare ops
        sess.run(model.initializing_ops)
        sess.run(model.prepare_ops)

        # And load variables if asked for.
        if args.checkpoint:
            tf_helpers.maching_variable_restore(sess, args.checkpoint)
            sess.run(model.final_ops)
            # trainer.saver.restore(sess, args.checkpoint)

        for epoch in range(1, args.e + 1):
            ####################################################################
            #### TRAIN
            ####################################################################
            trainer.train_step()

            ####################################################################
            #### VALIDATION
            ####################################################################
            if (epoch) % args.kv == 0:
                trainer.evaluation_step()


@args_debug
def test(args):
    chkp_folder = args.checkpoint.rsplit(os.path.sep, 1)[0]
    fn_config = os.path.join(chkp_folder, "config.json")

    with open(fn_config, 'r') as f:
        config = json.load(f)

    # Load Freebase dataset for now
    if config["dataset_type"] in datasets.__all__:
        dataset = getattr(datasets, config["dataset_type"])()
    else:
        dataset = datasets.BaseDataset(dataset=config["dataset_type"])
        # raise DatasetInvalid

    if config["model_type"] in models.__all__:
        try:
            model = getattr(models, config["model_type"])(dataset=dataset,
                                                          **config[
                                                              "model_configuration"])
        except TypeError as e:
            logger.error("Invalid Keyword/Argument supplied: ", e)
            raise ModelParametersInvalid
    else:
        raise ModelInvalid

    tf_config = generate_tf_config(args)

    with tf.Session(config=tf_config) as sess:

        # Wrap session by debug session
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        evaluator_config = args.evaluator_params
        evaluator_config["optimizer_settings"] = config["optimizer_settings"]
        evaluator_config["tf_config"] = tf_config
        evaluator_config["timeline"] = args.timeline
        evaluator_config["num_gpu"] = args.num_gpu
        evaluator_config["evaluation_mode"] = True

        evaluator = Trainer(dataset, model, sess, **evaluator_config)
        logger.debug(evaluator.config)

        # Setting up model via its init ops and follows prepare ops
        sess.run([model.init_ops])
        sess.run([model.prepare_ops])

        # Finally initialize everything else (e.g., optimizer)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Restore checkpoint
        tf_helpers.maching_variable_restore(sess, args.checkpoint)
        # trainer.saver.restore(sess, args.checkpoint)

        evaluator.evaluation_step(name="Evaluation")


if __name__ == "__main__":
    parser = ArgumentParser(prog='eknog',
                            description="Learning KG embeddings made easy.")
    subparsers = parser.add_subparsers()

    parser.add_argument('--debug', action='store_true',
                        help="Activate Tensorflow-Debugging.")

    # Training Mode
    train_cmd = subparsers.add_parser(name="train",
                                      help="Train a model (new or continue from saved checkpoint)")
    train_cmd.set_defaults(func=train)

    train_cmd.add_argument('--model', '-m', type=str, default="TransE",
                           help="Name of model.")
    train_cmd.add_argument('--dataset', '-d', type=str,
                           default="kinship",
                           help="Name of dataset.")
    train_cmd.add_argument('--model_params', '-mp', type=json.loads,
                           default="{}",
                           help="Model parameters as dict (json-formatted)")
    train_cmd.add_argument('--trainer_params', '-tp', type=json.loads,
                           default="{}",
                           help="Trainer parameters as dict (json-formatted)")
    train_cmd.add_argument('--optimizer_params', '-op', type=json.loads,
                           default='{}',
                           help="Trainer parameters as dict (json-formatted)")
    train_cmd.add_argument('--num_gpu', '-ng', default=-1, type=int,
                           # nargs='+', default=[],
                           help='How many GPUs to use. [Conflicts --cpu]')
    train_cmd.add_argument('--timeline', action='store_true',
                           help="Activate Full-Tracing options.")
    train_cmd.add_argument('--checkpoint', '-c', type=str, default="",
                           help="To continue from a previous checkpoint. Allows to load pretrained embeddings")
    train_cmd.add_argument('-kv', type=int, default=20,
                           help="Run validation every #kv epochs.")
    train_cmd.add_argument('-e', type=int, default=1000,
                           help="Run for #e epochs.")
    train_cmd.add_argument('--tag', type=str,
                           help="Place evaluation (checkpoints/tf_events) in subfolder <tag>.",
                           default='no_tag')

    # Testing Mode
    test_cmd = subparsers.add_parser(name="test",
                                     help="Run test set on saved model (from checkpoint)")
    test_cmd.set_defaults(func=test)
    test_cmd.add_argument('--evaluator_params', '-ep', type=json.loads,
                          default="{}",
                          help="Evaluator parameters as dict (json-formatted)")
    test_cmd.add_argument('--num_gpu', '-ng', default=1, type=int,
                          # nargs='+', default=[],
                          help='How many GPUs to use. [Conflicts --cpu]')
    test_cmd.add_argument('--timeline', action='store_true',
                          help="Activate Full-Tracing options.")
    test_cmd.add_argument('--checkpoint', '-c', type=str, default="",
                          help="Checkpoint/Model to evaluate.", required=True)
    test_cmd.add_argument('-kv', type=int, default=20)

    # Parse command-line arguments and run subroutines
    opt = parser.parse_args()

    # No arguments passed? Print help and quit, otherwise call subroutine
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)
    else:
        opt.func(opt)
