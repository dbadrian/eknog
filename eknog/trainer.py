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

import json
import os
import time
import logging

logger = logging.getLogger(__name__)


import tensorflow as tf
from tqdm import tqdm

import eknog.utils as common
import eknog.datasets as datasets
import eknog.eval_helpers as eval_helpers
import eknog.tf_helpers as tf_helpers
import eknog.tf_logging as tf_logging
from eknog.utils import generate_config_str


class Trainer():
    def __init__(self, dataset, model, session, epochs=1000,
                 train_batch_size=2048, validation_batch_size=512,
                 truncated_validation=-1, evaluation_mode=False,
                 validation_every_n_steps=-1, num_gpu=None,
                 optimizer="AdamOptimizer", optimizer_settings=None,
                 tf_config=None, timeline=False, max_checkpoints=20, tag=""):

        assert type(
            optimizer_settings) == dict, "Optimizer Settings not a of type dict!"

        self.dataset = dataset
        self.model = model
        # Compiles all initializers and prep ops
        self.model.finish_model()

        self.session = session

        self.config = {
            "tf_version": tf.__version__,
            "max_epochs": epochs,
            "model_type": self.model.__class__.__name__,
            "dataset_type": self.dataset.__class__.__name__,
            "train_batch_size": train_batch_size,
            "validation_batch_size": validation_batch_size,
            "truncated_validation": truncated_validation,
            "validation_n_steps": validation_every_n_steps,
            "optimizer": optimizer,
            "optimizer_settings": optimizer_settings,
            "model_configuration": model.config,
            "num_gpu": num_gpu,
            "proc_units": max(1, num_gpu)
        }

        self.dataset_sampler = datasets.DatasetSampler(self.dataset,
                                                       train_batch_size=
                                                       self.config[
                                                           "train_batch_size"],
                                                       evaluation_batch_size=
                                                       self.config[
                                                           "validation_batch_size"])

        # Folders for loggin and storing checkpoints
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fn_run = timestamp + "-" + generate_config_str(self.config)
        self.fn_tf_events = os.path.join('logs', tag, 'tf_events', fn_run)
        self.fn_checkpoints = os.path.join('logs', tag, 'checkpoints', fn_run)
        self.fn_checkpoint_prefix = os.path.join(self.fn_checkpoints, timestamp)
        common.mkdir_p(self.fn_checkpoints)
        with open(os.path.join(self.fn_checkpoints, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=4)

        # Create a saver object which will save all the variables
        self.saver = tf.train.Saver(var_list=self.model.var_list,
                                    max_to_keep=max_checkpoints,
                                    pad_step_number=True)

        # Tensorflow settings
        self.timeline = timeline
        if timeline:
            self.run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_options = None
            self.run_metadata = None

        self.tf_config = tf_config

        # Build all graphs
        if not evaluation_mode:
            self.__build_train_graph()
            self.__build_validation_graph(data_type='valid')
        else:
            self.__build_validation_graph(data_type='test')

        # Runtime variables
        self.epoch = 0
        self.processed_batches = 0
        if self.model.config["loss_type"] == 'BCE':
            self.config["total_training_batches"] = int(
                len(self.dataset_sampler.hr2e) / self.config[
                    "train_batch_size"])
        else:
            self.config["total_training_batches"] = int(
                self.dataset.number_of_entries["train"] / self.config[
                    "train_batch_size"])

        # This does not consider OOG, TODO: figure out correct ogg_n and add that
        val_total = self.dataset.number_of_entries[
            "valid"] if not evaluation_mode else self.dataset.number_of_entries[
            "test"]
        self.config["total_validation_batches"] = int(
            val_total / self.config["validation_batch_size"])

    def train_step(self):
        time_epoch = time.time()
        self.session.run(self.train_iterator_init_op)
        loss_epoch = 0
        batch = 1
        batch_times = []
        self.epoch += 1
        with tqdm(total=self.config["total_training_batches"],
                  desc="Epoch {}".format(self.epoch)) as pbar:
            while True:  # EPOCH
                try:
                    t_start = time.time()
                    loss_batch, _, train_summary_out = self.session.run(
                        [self.loss, self.train, self.train_summary],
                        options=self.run_options,
                        run_metadata=self.run_metadata)
                    self.train_summary_group.write(train_summary_out,
                                                   self.processed_batches)
                    self.processed_batches += self.config["proc_units"]

                    pbar.update(self.config["proc_units"])
                    # batch_times.append(time.time() - t_start)
                    loss_epoch += loss_batch

                    if self.timeline:
                        tf_helpers.write_timeline(self.run_metadata,
                                                  'timeline_train.json')

                    batch += self.config["proc_units"]

                except tf.errors.OutOfRangeError:
                    pbar.close()
                    logger.info(
                        'EPOCH: {}, loss: {}, total_time: {}'.format(
                            self.epoch,
                            loss_epoch / self.config["proc_units"],
                            time.time() - time_epoch))
                    break

        # If validaton batch-size is set, we perform evaluation
        if self.config["validation_n_steps"] >= 1:
            if self.epoch % self.config["validation_n_steps"] == 0:
                self.evaluation_step(name='Validation')

    def evaluation_step(self, name='Validation'):
        logger.info("###############################################################")
        logger.info("@ Starting {}".format(name))
        logger.info("###############################################################")

        for eval_name, eval_func, ops, iterators, prep_ops in self.validation_ops:
            logger.info(">>>Running eval mode: {}".format(eval_name))
            if prep_ops is not None:
                logger.info("Running prep_ops...")
                self.session.run(prep_ops)

            values_dict = eval_func(self.session, self.config, eval_name, ops,
                                    iterators)

            self.validation_summary_group.write(global_step=self.epoch,
                                                values_dict=values_dict)

        # Save model to checkpoint
        self.saver.save(self.session, save_path=self.fn_checkpoint_prefix,
                        global_step=self.epoch)

        logger.info("###############################################################")
        logger.info("@ Finished")
        logger.info("###############################################################")

    def __build_train_graph(self):
        default_device = '/gpu:0' if self.config["num_gpu"] == 1 else '/cpu:0'

        self.train_graph = tf.get_default_graph()  # tf.Graph()
        with self.train_graph.as_default(), tf.device(default_device):

            # Create training dataset and iterator + init_op
            activate_bce = True if self.model.config[
                                       "loss_type"] == "BCE" else False
            self.train_iterator = self.dataset_sampler.create_train_iterator(
                activate_bce=activate_bce)
            self.train_iterator_init_op = self.train_iterator.initializer

            # global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = 0.1
            # learning_rate = tf.train.exponential_decay(starter_learning_rate,
            #                                            global_step,
            #                                            100, 0.9, staircase=True)

            self.optimizer = getattr(tf.train, self.config["optimizer"])(
                **self.config["optimizer_settings"])

            if type(self.train_iterator.get_next()) != tuple:
                arguments = (self.train_iterator.get_next(),)
            else:
                arguments = self.train_iterator.get_next()

            if self.config["num_gpu"] <= 1:
                # Single GPU/soft-placement or CPU-only setup
                self.loss = self.model.loss(*arguments)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train = self.optimizer.minimize(self.loss,
                                                         var_list=self.model.trainable_vars)


            else:
                raise NotImplementedError(
                    "Multi-GPU Support not implemented right now!")

            self.train_summary_group = tf_logging.SummaryGroup(
                os.path.join(self.fn_tf_events, 'train'),
                graph=self.train_graph)
            self.train_summary_group.add_summary(
                tf.summary.scalar('loss', self.loss))
            self.train_summary = self.train_summary_group.merge_summaries()

    def __generate_eval_data_iterators(self, data_type):
        transe_head_iter, transe_rel_iter, transe_tail_iter = self.dataset_sampler.create_transe_evaluation_iterators(
            data_type=data_type)
        pr_head_iter, pr_rel_iter, pr_tail_iter = self.dataset_sampler.create_pr_evaluation_iterators(
            data_type=data_type)
        blob = {
            'TransE-Ranking': {
                'head': (transe_head_iter, transe_head_iter.initializer),
                'rel': (transe_rel_iter, transe_rel_iter.initializer),
                'tail': (transe_tail_iter, transe_tail_iter.initializer)
            },
            'PR': {
                'head': (pr_head_iter, pr_head_iter.initializer),
                'rel': (pr_rel_iter, pr_rel_iter.initializer),
                'tail': (pr_tail_iter, pr_tail_iter.initializer)
            }
        }

        return blob

    def __eval_op_from_definition(self, def_blob, dataset_type):
        # If the ranking function has specific requirements to what set
        # to test on

        if def_blob["eval_func"] == 'TransE-Ranking':
            efunc = eval_helpers.transe_ranking_evaluation
        else:
            raise NotImplementedError(
                "This eval function has not been implemented yet!")

        return (def_blob["name"], efunc,
                self.__generate_tower_evaluation_ops(
                    self.dataset_iterators[dataset_type][def_blob["eval_func"]][
                        "head"][0],
                    self.dataset_iterators[dataset_type][def_blob["eval_func"]][
                        "rel"][0],
                    self.dataset_iterators[dataset_type][def_blob["eval_func"]][
                        "tail"][0],
                    def_blob["func"],
                    def_blob["func_params"]),
                self.dataset_iterators[dataset_type], def_blob["prep_ops"])

    def __build_validation_graph(self, data_type='valid'):
        self.eval_graph = tf.get_default_graph()  # tf.Graph()

        default_device = '/gpu:0' if self.config["num_gpu"] == 1 else '/cpu:0'
        with self.eval_graph.as_default(), tf.device(default_device):
            # Iterator for the head and the tail dataset, and respective
            # initializers
            self.dataset_iterators = {
                data_type: self.__generate_eval_data_iterators(data_type)
            }
            if 'oog' in self.dataset.data_type2array:
                logger.info("Adding OOG-defs to graph")
                self.dataset_iterators[
                    "oog"] = self.__generate_eval_data_iterators("oog")

            # This will contain the ops run later by the evaluation function
            self.validation_ops = []

            # TODO: Remove this part of the code?
            # if self.config["num_gpu"] == -1:  # < 2:
            #     raise NotImplementedError("Please explicitly specify a GPU number >=1 for now!")
            # else:
            #     print(
            #         "Evaluation Graph: Using Multi-GPU setup, with %d GPUS" %
            #         self.config["num_gpu"])

            # Iterate over function blobs generated inside the model
            for func_blob in (
            self.model.validation_funcs if data_type == 'valid' else self.model.evaluation_funcs):
                test_set_type = data_type if func_blob["data_set"] is None else \
                func_blob["data_set"]

                self.validation_ops.append(
                    self.__eval_op_from_definition(func_blob, test_set_type))

            self.validation_summary_group = tf_logging.SummaryGroup(
                os.path.join(self.fn_tf_events, data_type),
                graph=self.eval_graph)

    def __generate_tower_evaluation_ops(self, head_iterator, rel_iterator,
                                        tail_iterator, rank_func,
                                        rank_func_params=None):
        ranks_head = []
        ranks_head_filtered = []
        ranks_rel = []
        ranks_rel_filtered = []
        ranks_tail = []
        ranks_tail_filtered = []

        if self.config["num_gpu"] <= 1:
            ops = self.__generate_ranking_evaluation_ops(head_iterator,
                                                         rel_iterator,
                                                         tail_iterator,
                                                         rank_func,
                                                         rank_func_params)

            ranks_head.append(ops["head"][0])
            ranks_head_filtered.append(ops["head"][1])
            ranks_rel.append(ops["rel"][0])
            ranks_rel_filtered.append(ops["rel"][1])
            ranks_tail.append(ops["tail"][0])
            ranks_tail_filtered.append(ops["tail"][1])
        else:
            # TODO: MultiGPU Deactivated
            raise NotImplementedError(
                "Multi-GPU Support not implemented right now!")
            # for gpu in range(self.config["num_gpu"]):
            #     with tf.device('/gpu:%d' % gpu):
            #         with tf.variable_scope('%s_tower-eval-%d' % (self.config["model_type"], gpu)) as scope:
            #
            #             ops =  self.__generate_ranking_evaluation_ops(head_iterator, rel_iterator, tail_iterator, rank_func, rank_func_params)
            #
            #             ranks_head.append(ops["head"][0])
            #             ranks_head_filtered.append(ops["head"][1])
            #             ranks_rel.append(ops["rel"][0])
            #             ranks_rel_filtered.append(ops["rel"][1])
            #             ranks_tail.append(ops["tail"][0])
            #             ranks_tail_filtered.append(ops["tail"][1])

        # Point of sync'ing
        return {
            "rank_head_op": tf.concat(ranks_head, axis=-1),
            "rank_head_filtered_op": tf.concat(ranks_head_filtered, axis=-1),
            "rank_rel_op": tf.concat(ranks_rel, axis=-1),
            "rank_rel_filtered_op": tf.concat(ranks_rel_filtered, axis=-1),
            "rank_tail_op": tf.concat(ranks_tail, axis=-1),
            "rank_tail_filtered_op": tf.concat(ranks_tail_filtered, axis=-1),
        }

    def __generate_ranking_evaluation_ops(self, head_iterator, rel_iterator,
                                          tail_iterator, rank_func,
                                          rank_func_params=None):
        """
        Create evaluation ops for ranking functions. A valid rank_func takes a variable for triplets and filtermask, as well as an entry-arg
        """
        head_data = head_iterator.get_next()
        rel_data = rel_iterator.get_next()
        tail_data = tail_iterator.get_next()

        # ops dict
        if rank_func_params is None:
            rank_func_params = {}

        return {
            "head": rank_func(*head_data, entry=0, **rank_func_params),
            "rel": rank_func(*rel_data, entry=1, **rank_func_params),
            "tail": rank_func(*tail_data, entry=2, **rank_func_params)
        }
