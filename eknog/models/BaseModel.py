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

import logging
from abc import ABC

import tensorflow as tf

import eknog.exceptions as ex
import eknog.tf_helpers as tf_helpers
import eknog.tf_ops
import eknog.tf_ops as tf_ops

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Every model used should inherit from this base class, as it contains
    all the important variables and functions the remaining code/trainer calls
    expects."""
    def __init__(self,
                 dataset,
                 embedding_dimension=100,
                 loss_type='softmax',
                 margin=1.0,
                 variable_device='/cpu:0',
                 var_dtype="float32"):

        if var_dtype != "float32":
            raise ValueError("Sorry, currently only float32 is supported")

        self.var_dtype = tf.as_dtype(var_dtype)

        # Sometimes we need to access stuff like constants (size of dataset etc.)
        self.dataset = dataset

        # To avoid typos when using kwargs, we can list allowed names here for
        # check at on initialization. Following lines depend on this!
        self.permissible_args = set()

        # All model args will be stored in this dict for logging.
        cleaned_args = self.__clean_copy_args(locals())
        self._add_permissible_args(cleaned_args)
        self.config = cleaned_args

        # For some models (e.g., TransE) we can have different scoring functions
        # (e.g., L1/L2), or distance measures. We can set that up here
        self.available_scoring_funcs = {}
        self.available_distance_funcs = {}

        # For some models we can have differing valdition and evaluation functions
        # In any case, they are stored here.
        self.validation_funcs = []
        self.evaluation_funcs = []

        # All tf variables are added to this list for storing/restoring etc.
        self.var_list = []

        # In some models, we want to dynamically disable the trainable property
        # When inheriting from a model, we can add variables here, and drop them
        # when computing gradients.
        self.blacklisted_train_vars = []

        # Variable initialization ops are added here
        self.init_ops = []

        # Some variable need special intialization steps and are added here
        self.prep_ops = []

        # Post-Restoring Ops. Run after a model has been loaded from disk, and
        # just before first training step.
        self.fin_ops = []

    def finish_model(self):
        """
        Compiles all variable initializing ops, preparation ops (more complex
        initializations), and post-restore operations. Additionally, it creates
        a list of all trainable variables in consideration of those added to the
        blacklist.
        """
        self.initializing_ops = tf.group(*tuple(self.init_ops))
        self.prepare_ops = tf.group(
            *tuple(self.prep_ops + [tf.tables_initializer()]))
        self.final_ops = tf.group(*tuple(self.fin_ops))

        self.trainable_vars = [var for var in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES) if
                               var not in self.blacklisted_train_vars]

        # And finally we will also check if there are any invalid args passed
        # to the model not listed in self.permissible_args
        print(self.config, self.permissible_args)
        for name, _ in self.config.items():
            if name not in self.permissible_args:
                raise ex.ModelParametersInvalid(
                    ">>> {} <<< was passed to the model, but this arg has not been defined!".format(
                        name))

    ############################################################################
    # Important functions for model inheritance:
    #   eknog supports that you don't need to redefine any of the arguments
    #   already defined in the parent model, but you can still give them to the
    #   derived model without problems.
    #   This was solved using some gnarly hacky stuff, see below
    ############################################################################
    def __clean_copy_args(self, args):
        """Removes some objects from given kwargs, which you never want here"""
        prune_list = ["self", "__class__"]
        return {key: val for key, val in args.items() if key not in prune_list}

    def _model_initialization(self, parent_class, args, define_model_func):
        """Utility function which should be called to correctly initialize any
        parent models (as more than just a super call is required."""
        if type(parent_class) == list:
            logger.warning("Deriving from multiple models is not tested yet! Unexpected behavior possible.")
            for pc in parent_class:
                self._filtered_super_call(pc.__init__,
                                          self.__clean_copy_args(args))
        else:
            self._filtered_super_call(parent_class.__init__,
                                      self.__clean_copy_args(args))

        self.config.update(
            self._generate_config_from_args(self.__clean_copy_args(args)))
        self._add_permissible_args(self.__clean_copy_args(args))
        define_model_func()

    def _filtered_super_call(self, init_func, args):
        """Makes a super call, but only with use arguments defined by the parent
        class."""
        valid_arg = init_func.__code__.co_varnames

        n_args = {key: args[key] for key in args if
                  key in valid_arg and key != "kwargs"}
        n_args.update(args["kwargs"])

        init_func(self, **n_args)

    def _add_permissible_args(self, args):
        """Creates a lookup table of args which can be given to a model. Will
        used later to verify args given to a model."""

        # Clean up the args
        args.pop('kwargs', None)
        args.pop("dataset", None)

        # Generate config
        config = self._generate_config_from_args(args)
        self.permissible_args.update(config.keys())

    def _add_init_op(self, ops):
        if type(ops) == list:
            self.init_ops += ops
        else:
            self.init_ops.append(ops)

    def _add_prep_op(self, ops):
        if type(ops) == list:
            self.prep_ops += ops
        else:
            self.prep_ops.append(ops)

    def _add_final_op(self, ops):
        if type(ops) == list:
            self.fin_ops += ops
        else:
            self.fin_ops.append(ops)

    def _generate_config_from_args(self, args):
        config = {}
        # If we reload a model for evaluation, we want to only consider the
        # config from files, else use args!
        if "config" not in args or args["config"] is None:

            # Set config to args
            config = args

            # merge kwargs into the config
            args.pop('config', None)
            kwargs = args.pop('kwargs', None)
            if kwargs is not None:
                config.update(kwargs)

            # Remove non-config/non-pickable items
            args.pop('self', None)
            args.pop('dataset', None)
            args.pop('__class__', None)
        else:
            config = args["config"]
        return config

    def _add_variable(self, name, shape, dtype, initializer, trainable,
                      device='/cpu:0', weight_decay=None, wd_func=tf.nn.l2_loss,
                      l2_normalize_axis=None, store_on_checkpoint=True):
        # Create variable with desired properties on specified device
        v = tf_helpers.variable_on_device(name=name, shape=shape, dtype=dtype,
                                          initializer=initializer,
                                          trainable=trainable,
                                          device=device,
                                          weight_decay=weight_decay,
                                          wd_func=wd_func)

        # Create a normalization operation if desired and add to list of prep ops
        if l2_normalize_axis is not None:
            normalize_op = tf.assign(v, tf.nn.l2_normalize(v,
                                                           dim=l2_normalize_axis))
            self._add_prep_op(normalize_op)

        # Add the initializer to init ops list
        self._add_init_op(v.initializer)

        # Only add it to the var_list (stored when saving checkpoints) if specified
        if store_on_checkpoint:
            self.var_list.append(v)

        return v

    def _blacklist_trainable_variable(self, var):
        self.blacklisted_train_vars.append(var)

    def _add_evaluation_func(self, name, func, eval_func, func_params=None,
                             prep_ops=None, data_set=None, evaluation=True,
                             validation=True):
        # Validate some of the inputs
        if prep_ops is not None:
            assert type(prep_ops) == list, "Preparation ops must be a list!"

        # Collect all args into a dict
        blob = locals()

        # Remove non-config items
        blob.pop('self')

        if validation:
            self.validation_funcs.append(blob)

        if evaluation:
            self.evaluation_funcs.append(blob)

    def _reset_evaluation_functions(self):
        del self.validation_funcs[:]
        del self.evaluation_funcs[:]

    def _generate_negative_samples(self, positive_triplets, k_negative_samples,
                                   sample_negative_relations,
                                   return_split_size=False, rel_ratio=0.5,
                                   n_relations=2):

        if sample_negative_relations:
            N_split = int(k_negative_samples / 3)
            N_r = min(N_split, int(rel_ratio * n_relations))
            N_split = int((k_negative_samples - N_r) / 2)
        else:
            N_split = int(k_negative_samples / 2)
            N_r = None

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(positive_triplets)

        # Sample K negative ("corrupted") samples per positive sample
        neg_head_ids = eknog.tf_ops.negative_sampling_uniform(positive_triplets,
                                                              N_split,
                                                              self.dataset.n_entities)
        if sample_negative_relations:
            if self.dataset.n_relations < 0.3 * N_split:
                neg_rel_ids = eknog.tf_ops.negative_sampling_uniform_with_exclusion(
                    rel_ids, N_r, self.dataset.n_relations)
            else:
                # this much cheaper, is if the number of relations is large, do this one
                neg_rel_ids = eknog.tf_ops.negative_sampling_uniform(
                    positive_triplets, N_r, self.dataset.n_relations)
        else:
            neg_rel_ids = None
        neg_tail_ids = eknog.tf_ops.negative_sampling_uniform(positive_triplets,
                                                              N_split,
                                                              self.dataset.n_entities)

        if not return_split_size:
            return neg_head_ids, neg_rel_ids, neg_tail_ids
        else:
            return neg_head_ids, neg_rel_ids, neg_tail_ids, N_split, N_r

    def _loss_out(self, scores, logit_mask=None):
        with tf.name_scope('loss') as scope:
            loss = 0

            # Add all weight decay variables to loss
            with tf.name_scope('weight_decay') as scope:
                wd_coll = tf.get_collection('weight_decay')
                if wd_coll:
                    loss += tf.add_n(wd_coll, name='total_wd')

            # Now process the list of energies
            if self.config["loss_type"] != "BCE":
                for score in scores:
                    if self.config["loss_type"] == "softplus":
                        # NOTE: Positive scores are multiplied by -1, causing the
                        # loss to approach zero for large positive scores.
                        loss += tf.reduce_sum(tf.nn.softplus(-1 * score[0],
                                                             name='positive_log_loss'))
                        loss += tf.reduce_sum(
                            tf.nn.softplus(score[1], name='negative_log_loss'))
                    elif self.config["loss_type"] == "softmax":
                        s0 = score[0]
                        total = tf.concat([s0, score[1]], -1)
                        max = tf.reduce_max(total, axis=-1, keepdims=True)
                        total = tf.exp(total - max, name="softmax_neg_exp")
                        nom = tf.exp(s0 - max, name="softmax_pos_exp")
                        denom = tf.reduce_sum(total, axis=-1, keepdims=True,
                                              name="softmax_neg_sum")
                        loss += -1 * tf.reduce_sum(tf.log(tf.div(nom, denom)),
                                                   name='softmax_loss_sum')
                    elif self.config["loss_type"] == "max_margin_loss":
                        loss += tf_ops.max_margin_loss(score[1] - score[0],
                                                       self.config["margin"],
                                                       name='max_margin_loss')
                    elif self.config["loss_type"] == "log":
                        total = tf.concat([score[0], score[1]], -1)
                        max = tf.reduce_max(total, axis=-1, keepdims=True)
                        e1 = score[1] - max
                        e0 = score[0] - max
                        loss += tf.reduce_sum(tf.log(1 + tf.exp(e1 - e0)))
                    elif self.config["loss_type"] == "ssl":
                        loss += tf.reduce_sum(tf.square(score[1]) + tf.square(
                            tf_ops.max_margin_loss(-1 * score[0],
                                                   self.config["margin"],
                                                   name='SSL')))
                    else:
                        raise NotImplementedError(
                            "This loss type has not been defined for {}".format(
                                self.__class__.__name__))
            else:
                # out = tf.sigmoid(scores)
                # label_smoothing = self.config["label_smoothing"] if "label_smoothing" in self.config else 0.0
                label_smoothing = 0.0
                logit_mask = tf.cast(logit_mask,
                                     dtype=tf.float32)  # according to the code, not necessary?
                # loss += tf.losses.sigmoid_cross_entropy(logit_mask, scores, label_smoothing=label_smoothing)

                # loss += -tf.reduce_sum(((logit_mask * tf.log(out + 1e-9)) + ((1 - logit_mask) * tf.log(1 - out + 1e-9))), name='xentropy')

                loss += tf.losses.sigmoid_cross_entropy(logit_mask, scores,
                                                        label_smoothing=label_smoothing)

            return loss

    def _generate_target_alternatives(self, test_triplets, entry, entity_lim,
                                      relation_lim):
        # Generate Tensor of the positive triplet, and all possible alternatives
        # Depends on whether we predict head/tails OR relations!
        test_targets = test_triplets[:, entry]
        if entry is 0 or entry is 2:
            alt_ids = tf.range(0, entity_lim)
        else:
            alt_ids = tf.range(0, relation_lim)

        test_targets = tf.expand_dims(test_targets, 1)
        return tf.concat([test_targets, alt_ids], 1)

    def _lookup_test_embs(self, test_triplets, test_targets, entry, entity_var,
                          relation_var, expand_dims=True):
        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(test_triplets)

        cargs = {"expand_dim": 1} if expand_dims else {}
        if entry == 0:  # Predict heads
            head_emb = tf_ops.emb_lookup(entity_var, test_targets)
            rel_emb = tf_ops.emb_lookup(relation_var, rel_ids, **cargs)
            tail_emb = tf_ops.emb_lookup(entity_var, tail_ids, **cargs)
        elif entry == 1:  # Predict relations
            head_emb = tf_ops.emb_lookup(entity_var, head_ids, **cargs)
            rel_emb = tf_ops.emb_lookup(relation_var, test_targets)
            tail_emb = tf_ops.emb_lookup(entity_var, tail_ids, **cargs)
        else:  # Predict tails
            head_emb = tf_ops.emb_lookup(entity_var, head_ids, **cargs)
            rel_emb = tf_ops.emb_lookup(relation_var, rel_ids, **cargs)
            tail_emb = tf_ops.emb_lookup(entity_var, test_targets)

        return head_emb, rel_emb, tail_emb

    def _rank_by_score(self, score, filter_mask, ascending_order=True):
        # TODO replace number by import sys/sys.float_info.max -> inf
        filter_mask = tf.cast(filter_mask, dtype=tf.float32) * 999999999999999.0
        if ascending_order:
            score_filtered = tf.add(score[1:], filter_mask)
        else:
            score_filtered = tf.subtract(score[1:], filter_mask)

        rank = tf_ops.rank_array_by_first_element(score,
                                                  reverse=not ascending_order)
        rank_filtered = tf_ops.rank_array_by_first_element(score_filtered,
                                                           reverse=not ascending_order)

        return rank, rank_filtered

    def _rank_by_score_with_pivots(self, scores, pivot_ids, filter_mask,
                                   ascending_order=True, return_scores=False):
        # TODO replace number by import sys/sys.float_info.max -> inf
        filter_mask = tf.cast(filter_mask, dtype=tf.float32) * 999999999999999.0

        batch_size = tf.shape(scores)[0]
        pivot_ids_x = tf.expand_dims(pivot_ids, 1)
        pivot_ids_x = tf.concat(
            [tf.expand_dims(tf.range(0, batch_size), 1), pivot_ids_x], 1)

        pivot_scores = tf.expand_dims(tf.gather_nd(scores, pivot_ids_x), 1)

        if ascending_order:
            scores_filtered = tf.add(scores, filter_mask)
        else:
            scores_filtered = tf.subtract(scores, filter_mask)

        rank = tf_ops.rank_array_by_pivots(scores, pivot_scores,
                                           reverse=not ascending_order)
        rank_filtered = tf_ops.rank_array_by_pivots(scores_filtered,
                                                    pivot_scores,
                                                    reverse=not ascending_order)

        if return_scores:
            return rank, rank_filtered, scores, scores_filtered
        else:
            return rank, rank_filtered

    def _add_scoring_function(self, name, func):
        self.available_scoring_funcs[name] = func

    def _set_scoring_function(self, name):
        if name in self.available_scoring_funcs:
            self.score_func = self.available_scoring_funcs[name]
        else:
            raise NotImplementedError(
                name + " has not been implemented for this model!")

    def _add_distance_function(self, name, func):
        self.available_distance_funcs[name] = func

    def _set_distance_function(self, name):
        if name in self.available_distance_funcs:
            self.distance_func = self.available_distance_funcs[name]
        else:
            raise NotImplementedError(
                name + " has not been implemented for this model!")
