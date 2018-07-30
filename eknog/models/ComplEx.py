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


class ComplEx(BaseModel):
    # TODO: ADD ORIGINAL PAPER LINK, DESCRIBE MODEL IN FEW WORDS
    def __init__(self, initialize_uniform=False, emb_dropout=0.0,
                 entity_wd=None, entity_wd_type="L2",
                 rel_wd=None, rel_wd_type="L2",
                 neg_rel_ratio=0.5,
                 sample_negative_relations=False, k_negative_samples=30,
                 **kwargs):

        self._model_initialization(BaseModel, locals(), self.__define_model)

    def __define_model(self):
        initializer = tf.contrib.layers.xavier_initializer(
            self.config["initialize_uniform"])
        self.score_func = lambda arg0_r, arg0_i, arg1_r, arg1_i, arg2_r, arg2_i, \
                                 shape_right, n_dim=None, \
                                 arg_right=2: tf_ops.complex_trilinear_dot_product(
            arg0_r, arg0_i, arg1_r, arg1_i, arg2_r, arg2_i,
            shape_right=shape_right, n_dim=n_dim,
            d_dim=self.config["embedding_dimension"], arg_right=arg_right)

        self.score_func = tf_ops.complex_trilinear_dot_product

        variable_init_params = {
            "dtype": self.var_dtype,
            "device": self.config["variable_device"],
            "initializer": initializer,
            "trainable": True
        }

        entity_wd_func = tf_ops.select_norm_by_string(
            self.config["entity_wd_type"])
        rel_wd_func = tf_ops.select_norm_by_string(self.config["rel_wd_type"])

        with tf.name_scope("embeddings"):
            self.entity_embeddings_real = self._add_variable(
                name="entity_embeddings_real",
                shape=[self.dataset.n_entities,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["entity_wd"], wd_func=entity_wd_func,
                **variable_init_params)

            self.entity_embeddings_imag = self._add_variable(
                name="entity_embeddings_imag",
                shape=[self.dataset.n_entities,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["entity_wd"], wd_func=entity_wd_func,
                **variable_init_params)

            self.relation_embeddings_real = self._add_variable(
                name="relation_embeddings_real",
                shape=[self.dataset.n_relations,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["rel_wd"], wd_func=rel_wd_func,
                **variable_init_params)

            self.relation_embeddings_imag = self._add_variable(
                name="relation_embeddings_imag",
                shape=[self.dataset.n_relations,
                       self.config["embedding_dimension"]],
                weight_decay=self.config["rel_wd"], wd_func=rel_wd_func,
                **variable_init_params)

        self._add_evaluation_func("Structure-Ranks", self.rank,
                                  'TransE-Ranking')

    def loss(self, positive_triplets):

        energies = []
        pos_scores = []
        neg_scores = []

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(positive_triplets)

        with tf.name_scope('positive_emb_lookup') as scope:
            # Get scores for positive test examples (Y_rso==1)
            pos_head_emb_real = tf_ops.emb_lookup(self.entity_embeddings_real,
                                                  head_ids)
            pos_head_emb_imag = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                                  head_ids)
            pos_rel_emb_real = tf_ops.emb_lookup(self.relation_embeddings_real,
                                                 rel_ids)
            pos_rel_emb_imag = tf_ops.emb_lookup(self.relation_embeddings_imag,
                                                 rel_ids)
            pos_tail_emb_real = tf_ops.emb_lookup(self.entity_embeddings_real,
                                                  tail_ids)
            pos_tail_emb_imag = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                                  tail_ids)

            pos_head_emb_real, pos_head_emb_imag = tf_ops.complex_dropout(
                pos_head_emb_real, pos_head_emb_imag,
                self.config["emb_dropout"])
            pos_rel_emb_real, pos_rel_emb_imag = tf_ops.complex_dropout(
                pos_rel_emb_real, pos_rel_emb_imag, self.config["emb_dropout"])
            pos_tail_emb_real, pos_tail_emb_imag = tf_ops.complex_dropout(
                pos_tail_emb_real, pos_tail_emb_imag,
                self.config["emb_dropout"])

        with tf.name_scope('positive_score') as scope:
            # PAY ATTENTION TO THE DIFFERING ORDER RELATION->HEAD->TAIL!
            positive_scores = self.score_func(pos_rel_emb_real,
                                              pos_rel_emb_imag,
                                              pos_head_emb_real,
                                              pos_head_emb_imag,
                                              pos_tail_emb_real,
                                              pos_tail_emb_imag, 'bd')
            pos_scores.append(positive_scores)

        with tf.name_scope('negative_sampling') as scope:
            neg_head_ids, neg_rel_ids, neg_tail_ids, n_dim, n_r = self._generate_negative_samples(
                positive_triplets, self.config["k_negative_samples"],
                self.config["sample_negative_relations"],
                return_split_size=True, n_relations=self.dataset.n_relations,
                rel_ratio=self.config["neg_rel_ratio"])

        with tf.name_scope('negative_emb_lookup') as scope:
            neg_head_emb_real = tf_ops.emb_lookup(self.entity_embeddings_real,
                                                  neg_head_ids)
            neg_head_emb_imag = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                                  neg_head_ids)
            neg_tail_emb_real = tf_ops.emb_lookup(self.entity_embeddings_real,
                                                  neg_tail_ids)
            neg_tail_emb_imag = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                                  neg_tail_ids)

            neg_head_emb_real, neg_head_emb_imag = tf_ops.complex_dropout(
                neg_head_emb_real, neg_head_emb_imag,
                self.config["emb_dropout"])
            neg_tail_emb_real, neg_tail_emb_imag = tf_ops.complex_dropout(
                neg_tail_emb_real, neg_tail_emb_imag,
                self.config["emb_dropout"])

        with tf.name_scope('negative_scores') as scope:
            negative_scores_head = self.score_func(pos_rel_emb_real,
                                                   pos_rel_emb_imag,
                                                   neg_head_emb_real,
                                                   neg_head_emb_imag,
                                                   pos_tail_emb_real,
                                                   pos_tail_emb_imag, 'bnd',
                                                   n_dim=n_dim, arg_right=1)
            neg_scores.append(negative_scores_head)

            negative_scores_tail = self.score_func(pos_rel_emb_real,
                                                   pos_rel_emb_imag,
                                                   pos_head_emb_real,
                                                   pos_head_emb_imag,
                                                   neg_tail_emb_real,
                                                   neg_tail_emb_imag, 'bnd',
                                                   n_dim=n_dim, arg_right=2)
            neg_scores.append(negative_scores_tail)

            if self.config["sample_negative_relations"]:
                neg_rel_emb_real = tf_ops.emb_lookup(
                    self.relation_embeddings_real, neg_rel_ids)
                neg_rel_emb_imag = tf_ops.emb_lookup(
                    self.relation_embeddings_imag, neg_rel_ids)

                neg_rel_emb_real, neg_rel_emb_imag = tf_ops.complex_dropout(
                    neg_rel_emb_real, neg_rel_emb_imag,
                    self.config["emb_dropout"])

                negative_scores_rel = self.score_func(neg_rel_emb_real,
                                                      neg_rel_emb_imag,
                                                      pos_head_emb_real,
                                                      pos_head_emb_imag,
                                                      pos_tail_emb_real,
                                                      pos_tail_emb_imag, 'bnd',
                                                      n_dim=n_r, arg_right=0)
                neg_scores.append(negative_scores_rel)

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
            rel_emb_r = tf_ops.emb_lookup(self.relation_embeddings_real,
                                          rel_ids)
            rel_emb_i = tf_ops.emb_lookup(self.relation_embeddings_imag,
                                          rel_ids)
            tail_emb_r = tf_ops.emb_lookup(self.entity_embeddings_real,
                                           tail_ids)
            tail_emb_i = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                           tail_ids)
            scores = self.score_func(rel_emb_r, rel_emb_i,
                                     self.entity_embeddings_real,
                                     self.entity_embeddings_imag, tail_emb_r,
                                     tail_emb_i, 'nd',
                                     n_dim=self.dataset.n_entities, arg_right=1)
            pivots = head_ids
        elif entry == 1:
            head_emb_r = tf_ops.emb_lookup(self.entity_embeddings_real,
                                           head_ids)
            head_emb_i = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                           head_ids)
            tail_emb_r = tf_ops.emb_lookup(self.entity_embeddings_real,
                                           tail_ids)
            tail_emb_i = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                           tail_ids)
            scores = self.score_func(self.relation_embeddings_real,
                                     self.relation_embeddings_imag, head_emb_r,
                                     head_emb_i, tail_emb_r, tail_emb_i, 'nd',
                                     n_dim=self.dataset.n_relations,
                                     arg_right=0)
            pivots = rel_ids
        elif entry == 2:
            head_emb_r = tf_ops.emb_lookup(self.entity_embeddings_real,
                                           head_ids)
            head_emb_i = tf_ops.emb_lookup(self.entity_embeddings_imag,
                                           head_ids)
            rel_emb_r = tf_ops.emb_lookup(self.relation_embeddings_real,
                                          rel_ids)
            rel_emb_i = tf_ops.emb_lookup(self.relation_embeddings_imag,
                                          rel_ids)
            scores = self.score_func(rel_emb_r, rel_emb_i, head_emb_r,
                                     head_emb_i, self.entity_embeddings_real,
                                     self.entity_embeddings_imag, 'nd',
                                     n_dim=self.dataset.n_entities, arg_right=2)
            pivots = tail_ids

        return self._rank_by_score_with_pivots(scores, pivots, filter_mask,
                                               ascending_order=False)
