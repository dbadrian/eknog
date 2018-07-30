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


class TorusE(BaseModel):
    def __init__(self, margin=1.0, distance_measure="L2",
                 loss_type="max_margin_loss",
                 sample_negative_relations=False,
                 initialize_uniform=True, k_negative_samples=1, **kwargs):

        self._model_initialization(BaseModel, locals(), self.__define_model)

    def __define_model(self):

        self._add_distance_function("L1", tf_ops.distance_l1_torus)
        self._add_scoring_function("L1", tf_ops.score_l1_torus)

        self._add_distance_function("L2", tf_ops.distance_l2_torus)
        self._add_scoring_function("L2", tf_ops.score_l2_torus)

        self._add_distance_function("EXPL2", tf_ops.distance_exp_l2_torus)
        self._add_scoring_function("EXPL2", tf_ops.score_exp_l2_torus)

        # Since its TransE-type model, score function would be the distance
        # between h+r/t, here with projection to the Torusnormalize
        self._set_scoring_function(self.config["distance_measure"])
        self._set_distance_function(self.config["distance_measure"])

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
                shape=[
                    self.dataset.n_entities,
                    self.config["embedding_dimension"]],
                **variable_init_params)

            self.relation_embeddings = self._add_variable(
                name="relation_embeddings",
                shape=[self.dataset.n_relations,
                       self.config["embedding_dimension"]],
                **variable_init_params)

        self._add_evaluation_func("Structure-Ranks", self.rank,
                                  'TransE-Ranking')

    def predict_head(self, rel_emb, tail_emb):
        """Will calculate missing heads from batch of triplets (2D Tensors)"""
        return tail_emb - rel_emb

    def predict_relation(self, head_emb, tail_emb):
        """Will calculate missing rels from batch of triplets (2D Tensors)"""
        return tail_emb - head_emb

    def predict_tail(self, head_emb, rel_emb):
        """Will calculate missing tails from batch of triplets (2D Tensors)"""
        return head_emb + rel_emb

    def loss(self, positive_triplets):
        """
        Returns a loss variable which can be optimized.
        :param positive_triplets: A batch (2D Tensor) of positive triplets.
        """
        energies = []

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(positive_triplets)

        with tf.name_scope('negative_sampling') as scope:
            neg_head_ids, neg_rel_ids, neg_tail_ids = self._generate_negative_samples(
                positive_triplets, self.config["k_negative_samples"],
                self.config["sample_negative_relations"])

        project_entities = tf.assign(self.entity_embeddings,
                                     tf_ops.project_to_unit_tensor(
                                         self.entity_embeddings))
        project_relations = tf.assign(self.relation_embeddings,
                                      tf_ops.project_to_unit_tensor(
                                          self.relation_embeddings))

        with tf.control_dependencies([project_entities, project_relations]):
            with tf.name_scope('positive_sample_embeddings') as scope:
                pos_head_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                                 head_ids, expand_dim=1)
                pos_rel_emb = tf_ops.emb_lookup(self.relation_embeddings,
                                                rel_ids, expand_dim=1)
                pos_tail_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                                 tail_ids, expand_dim=1)

            with tf.name_scope('negative_sample_embeddings') as scope:
                neg_head_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                                 neg_head_ids)
                neg_tail_emb = tf_ops.emb_lookup(self.entity_embeddings,
                                                 neg_tail_ids)
                if self.config["sample_negative_relations"]:
                    neg_rel_emb = tf_ops.emb_lookup(self.relation_embeddings,
                                                    neg_rel_ids)

            with tf.name_scope('predictions') as scope:
                hr_t_emb = self.predict_tail(pos_head_emb, pos_rel_emb)
                hcr_t_emb = self.predict_tail(neg_head_emb, pos_rel_emb)
                if self.config["sample_negative_relations"]:
                    hrc_t_emb = self.predict_tail(pos_head_emb, neg_rel_emb)

            with tf.name_scope('distances') as scope:
                dist_hr_t = self.distance_func(hr_t_emb, pos_tail_emb, axis=-1)
                dist_hcr_t = self.distance_func(hcr_t_emb, pos_tail_emb,
                                                axis=-1)
                dist_hr_tc = self.distance_func(hr_t_emb, neg_tail_emb, axis=-1)
                if self.config["sample_negative_relations"]:
                    dist_hrc_t = self.distance_func(hrc_t_emb, pos_tail_emb,
                                                    axis=-1)

            with tf.name_scope('scores') as scope:
                score_hr_t = self.score_func(dist_hr_t)
                score_hcr_t = self.score_func(dist_hcr_t)
                score_hr_tc = self.score_func(dist_hr_tc)
                if self.config["sample_negative_relations"]:
                    score_hrc_t = self.score_func(dist_hrc_t)

            with tf.name_scope('energies') as scope:
                energies.append((score_hr_t, score_hcr_t))
                energies.append((score_hr_t, score_hr_tc))
                if self.config["sample_negative_relations"]:
                    energies.append((score_hr_t, score_hrc_t))

        if self.config["loss_type"] == "softmax" or self.config[
            "loss_type"] == "softplus":
            neng = []
            for pos, neg in energies:
                neng.append((-1 * pos, -1 * neg))
            return self._loss_out(neng)
        else:
            return self._loss_out(energies)

    def rank(self, test_triplets, filter_mask, entry):
        assert entry == 0 or entry == 1 or entry == 2, "Entry is not {0,1,2}."

        head_ids, rel_ids, tail_ids = tf_ops.split_triplet(test_triplets)

        target_emb = self.relation_embeddings if entry == 1 else self.entity_embeddings
        if entry == 0:
            rel_emb = tf_ops.emb_lookup(self.relation_embeddings, rel_ids)
            tail_emb = tf_ops.emb_lookup(self.entity_embeddings, tail_ids)
            predicted = self.predict_head(rel_emb, tail_emb)
            pivots = head_ids
        elif entry == 1:
            head_emb = tf_ops.emb_lookup(self.entity_embeddings, head_ids)
            tail_emb = tf_ops.emb_lookup(self.entity_embeddings, tail_ids)
            predicted = self.predict_relation(head_emb, tail_emb)
            pivots = rel_ids
        elif entry == 2:
            head_emb = tf_ops.emb_lookup(self.entity_embeddings, head_ids)
            rel_emb = tf_ops.emb_lookup(self.relation_embeddings, rel_ids)
            predicted = self.predict_tail(head_emb, rel_emb)
            pivots = tail_ids

        """A special case arises, and is also supported, where each of the input
        arrays has a degenerate dimension at a different index. In this case, 
        the result is an "outer operation": (2,1) and (1,3) broadcast to (2,3).
        For more examples, consult the Numpy documentation on broadcasting."""
        predicted = tf.expand_dims(predicted, 1)
        target_emb = tf.expand_dims(target_emb, 0)
        # Calculate distance
        distances = self.distance_func(target_emb, predicted, axis=-1)
        scores = self.score_func(distances)

        return self._rank_by_score_with_pivots(scores, pivots, filter_mask,
                                               ascending_order=True)
