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

import tensorflow as tf

import eknog.tf_ops as tf_ops
from . import BaseModel

logger = logging.getLogger(__name__)


class Analogy(BaseModel):
    def __init__(self,
                 wd=None, wd_type="L2",
                 label_smoothing=0.0,
                 neg_rel_ratio=0.5,
                 num_scalar=100,
                 sample_negative_relations=False, emb_dropout=0.0,
                 variable_device='/cpu:0',
                 initialize_uniform=True, k_negative_samples=1, **kwargs):

        self._model_initialization(BaseModel, locals(), self.__define_model)

    def __define_model(self):
        # For reuse in child-models
        # self.score_func = lambda arg0, arg1, arg2, shape_right, n_dim=None, arg_right=2: tf_ops.trilinear_dot_product(arg0,
        #                                                                    arg1,
        #                                                                    arg2,
        #                                                                    shape_right=shape_right,
        #                                                                    n_dim=n_dim,
        #                                                                    d_dim=
        #                                                                    self.config[
        #                                                                        "embedding_dimension"],
        #                                                                    arg_right=arg_right)

        self.loss = self._standard_loss

        wd_func = tf_ops.select_norm_by_string(
            self.config["wd_type"])

        # These are basically mandatory variables
        with tf.name_scope("embeddings"):
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=self.config["initialize_uniform"])

            variable_init_params = {
                "dtype": self.var_dtype,
                "device": self.config["variable_device"],
                "initializer": initializer,
                "trainable": True
            }

            self.entity_embeddings = self._add_variable(
                name="entity_embeddings",
                shape=[self.dataset.n_entities,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["wd"], wd_func=wd_func,
                **variable_init_params)

            self.relation_embeddings = self._add_variable(
                name="relation_embeddings",
                shape=[self.dataset.n_relations,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["wd"], wd_func=wd_func,
                **variable_init_params)

        self._add_evaluation_func("Structure-Ranks", self.rank,
                                  'TransE-Ranking')


    def score_func(self, arg0, arg1, arg2, shape_right, n_dim=None, arg_right=2):
        l = [arg0, arg1, arg2]
        args = []
        for idx, arg in enumerate(l):
            args.append(self.partition_var(l[idx], self.config["num_scalar"], self.config["embedding_dimension"], len(shape_right) if idx == arg_right else 2))

        return self.score_func_int(args[0][0], args[1][0], args[2][0],
                                   args[0][1], args[0][2],
                                   args[1][1], args[1][2],
                                   args[2][1], args[2][2], shape_right, n_dim, arg_right)


    def partition_var(self, arg, dim_top, dim_total, depth=2):
        if depth==3:
            return arg[:, :, :dim_top], arg[:, :, dim_top:int(dim_top+(dim_total - dim_top)/2)], arg[:, :, int(dim_top+(dim_total - dim_top)/2):]
        elif depth==2:
            return arg[:, :dim_top], arg[:, dim_top:int(dim_top+(dim_total - dim_top)/2)], arg[:,  int(dim_top+(dim_total - dim_top)/2):]



    def score_func_int(self,
                   sa0, sa1, sa2,
                   ba00, ba01,
                   ba10, ba11,
                   ba20, ba21, shape_right, n_dim=None, arg_right=2):
        dim2 = (self.config["embedding_dimension"] - self.config["num_scalar"])/2,
        dim2 = dim2[0]
        sym_dot = tf_ops.trilinear_dot_product(sa0, sa1, sa2, shape_right=shape_right, n_dim=n_dim, d_dim=self.config["num_scalar"], arg_right=arg_right)
        dot1 = tf_ops.trilinear_dot_product(ba00, ba10, ba20, shape_right=shape_right, n_dim=n_dim, d_dim=dim2, arg_right=arg_right)
        dot2 = tf_ops.trilinear_dot_product(ba00, ba11, ba21, shape_right=shape_right, n_dim=n_dim, d_dim=dim2, arg_right=arg_right)
        dot3 = tf_ops.trilinear_dot_product(ba01, ba11, ba20, shape_right=shape_right, n_dim=n_dim, d_dim=dim2, arg_right=arg_right)
        dot4 = tf_ops.trilinear_dot_product(ba01, ba10, ba21, shape_right=shape_right, n_dim=n_dim, d_dim=dim2, arg_right=arg_right)
        return sym_dot + dot1 - dot2 + dot3 + dot4

    def _standard_loss(self, positive_triplets):
        """
        Returns a loss variable which can be optimized.
        :param positive_triplets: A batch (2D Tensor) of positive triplets.
        """
        energies = []
        pos_scores = []
        neg_scores = []

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(positive_triplets)

        with tf.name_scope('positive_score') as scope:
            # Get scores for positive test examples (Y_rso==1)
            cargs = {"dropout": 1 - self.config["emb_dropout"]}

            pos_head_emb = tf_ops.emb_lookup(self.entity_embeddings, head_ids,
                                             **cargs)
            pos_rel_emb = tf_ops.emb_lookup(self.relation_embeddings, rel_ids,
                                            **cargs)
            pos_tail_emb = tf_ops.emb_lookup(self.entity_embeddings, tail_ids,
                                             **cargs)

            positive_scores = self.score_func(pos_head_emb, pos_rel_emb,
                                              pos_tail_emb, 'bd')
            pos_scores.append(positive_scores)

        with tf.name_scope('negative_sampling') as scope:
            neg_head_ids, neg_rel_ids, neg_tail_ids, n_dim, n_r = self._generate_negative_samples(
                positive_triplets, self.config["k_negative_samples"],
                self.config["sample_negative_relations"],
                return_split_size=True, n_relations=self.dataset.n_relations,
                rel_ratio=self.config["neg_rel_ratio"])

        with tf.name_scope('negative_emb_lookup') as scope:
            cargs = {"dropout": 1 - self.config["emb_dropout"]}
            neg_head_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                             neg_head_ids, **cargs)
            neg_tail_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                             neg_tail_ids, **cargs)

            if self.config["sample_negative_relations"]:
                neg_rel_emb = tf_ops.emb_lookup(self.relation_embeddings,
                                                neg_rel_ids, **cargs)

        with tf.name_scope('negative_scores') as scope:
            score_hcr_t = self.score_func(neg_head_emb, pos_rel_emb,
                                          pos_tail_emb, 'bnd', n_dim=n_dim,
                                          arg_right=0)
            score_hr_tc = self.score_func(pos_head_emb, pos_rel_emb,
                                          neg_tail_emb, 'bnd', n_dim=n_dim,
                                          arg_right=2)
            neg_scores.append(score_hcr_t)
            neg_scores.append(score_hr_tc)

            if self.config["sample_negative_relations"]:
                score_hrc_t = self.score_func(pos_head_emb, neg_rel_emb,
                                              pos_tail_emb, 'bnd', n_dim=n_r,
                                              arg_right=1)
                neg_scores.append(score_hrc_t)

        if self.config["loss_type"] in ["softplus"]:
            # pos_scores = tf.concat(pos_scores, axis=-1)
            neg_scores = tf.concat(neg_scores, axis=-1)
            energies.append((positive_scores, neg_scores))
        else:
            for ns in neg_scores:
                energies.append((positive_scores, ns))

        return self._loss_out(energies)

    def rank(self, test_triplets, filter_mask, entry):
        """
        Processes a batch of test-triplets and ranks the positive triplet's
        entry (head, relation, or tail) accordingly against all other entities
        in the datasat.

        Returns a raw rank and a filtered rank (with other correct entities
        removed, as they might rank before the positive target entry).
        """
        assert entry == 0 or entry == 1 or entry == 2, "Entry is not {0,1,2}."

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(test_triplets)

        if entry == 0:
            rel_emb = tf_ops.emb_lookup(self.relation_embeddings, rel_ids)
            tail_emb = tf_ops.emb_lookup(self.entity_embeddings, tail_ids)
            scores = self.score_func(self.entity_embeddings, rel_emb, tail_emb,
                                     'nd', n_dim=self.dataset.n_entities,
                                     arg_right=0)
            pivots = head_ids
        elif entry == 1:
            head_emb = tf_ops.emb_lookup(self.entity_embeddings, head_ids)
            tail_emb = tf_ops.emb_lookup(self.entity_embeddings, tail_ids)
            scores = self.score_func(head_emb, self.relation_embeddings,
                                     tail_emb, 'nd',
                                     n_dim=self.dataset.n_relations,
                                     arg_right=1)
            pivots = rel_ids
        elif entry == 2:
            head_emb = tf_ops.emb_lookup(self.entity_embeddings, head_ids)
            rel_emb = tf_ops.emb_lookup(self.relation_embeddings, rel_ids)
            scores = self.score_func(head_emb, rel_emb, self.entity_embeddings,
                                     'nd', n_dim=self.dataset.n_entities,
                                     arg_right=2)
            pivots = tail_ids

        return self._rank_by_score_with_pivots(scores, pivots, filter_mask,
                                               ascending_order=False)
