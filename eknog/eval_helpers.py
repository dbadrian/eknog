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
# =========================================================================

import time
import logging

logger = logging.getLogger(__name__)


import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Placeholder messages
msg_validation_mrr_raw = "Mean Rank (raw): {}, Mean Reciprocal Rank (raw): {}"
msg_validation_mrr_filtered = "Mean Rank (filtered): {}, Mean Reciprocal Rank (filtered): {}"
msg_validation_hits = "hits@1: {}, hits@3: {}, hits@10: {}"


def hits_at_k(ranks, k):
    """
    Calculates the hits@k metric (percentage of ranks <= k).
    It assumes that the lowest (best) rank starts at 0.
    :param ranks:
    :type ranks:
    :param k:
    :type k:
    :return:
    :rtype:
    """
    return 100.0 * np.sum(np.less(ranks, k + 1), dtype=np.float32) / len(ranks)


def mean_rank(ranks):
    """
    Calculates the Mean Rank (MR) metric.
    :param ranks:
    :type ranks:
    :return:
    :rtype:
    """
    return np.mean(ranks)


def mean_reciprocal_rank(ranks):
    """
    Calculates the Mean Rank (MR) metric.
    :param ranks:
    :type ranks:
    :return:
    :rtype:
    """
    return np.mean(np.reciprocal(np.array(ranks, dtype=np.float32)))


def transe_ranking_evaluation(session, config, eval_name, ops, iterators):
    """Calculates the TransE style evaluation (see Bordes et al.) by replacing
    head/rels/or tails and ranking against all alternatives (raw/filtered setting)."""
    logger.info(">>> head")
    head_ranks, head_ranks_filtered, head_proc_time = __transe_ranking_template(
        session, config,
        iterators["TransE-Ranking"]["head"][1], ops["rank_head_op"],
        ops["rank_head_filtered_op"], "Head-{}".format(eval_name))
    logger.info(">>> tail")
    tail_ranks, tail_ranks_filtered, tail_proc_time = __transe_ranking_template(
        session, config,
        iterators["TransE-Ranking"]["tail"][1], ops["rank_tail_op"],
        ops["rank_tail_filtered_op"], "Tail-{}".format(eval_name))
    logger.info(">>> rel")
    rel_ranks, rel_ranks_filtered, rel_proc_time = __transe_ranking_template(
        session, config,
        iterators["TransE-Ranking"]["rel"][1], ops["rank_rel_op"],
        ops["rank_rel_filtered_op"], "Rel-{}".format(eval_name))

    entity_ranks = head_ranks + tail_ranks
    entity_ranks_filtered = head_ranks_filtered + tail_ranks_filtered

    values_dict = {
        "Entity_MR_{}".format(eval_name): mean_rank(entity_ranks),
        "Entity_MRR_{}".format(eval_name): mean_reciprocal_rank(
            entity_ranks),
        "Entity_MR_(filtered)_{}".format(eval_name): mean_rank(
            entity_ranks_filtered),
        "Entity_MRR_(filtered)_{}".format(
            eval_name): mean_reciprocal_rank(
            entity_ranks_filtered),
        "Entity_hits_at_1_{}".format(eval_name): hits_at_k(
            entity_ranks_filtered, 1),
        "Entity_hits_at_3_{}".format(eval_name): hits_at_k(
            entity_ranks_filtered, 3),
        "Entity_hits_at_10_{}".format(eval_name): hits_at_k(
            entity_ranks_filtered, 10),
        "Relation_MR_{}".format(eval_name): mean_rank(rel_ranks),
        "Relation_MRR_{}".format(eval_name): mean_reciprocal_rank(
            rel_ranks),
        "Relation_MR_(filtered)_{}".format(eval_name): mean_rank(
            rel_ranks_filtered),
        "Relation_MRR_(filtered)_{}".format(
            eval_name): mean_reciprocal_rank(rel_ranks_filtered),
        "Relation_hits_at_1_{}".format(eval_name): hits_at_k(
            rel_ranks_filtered, 1),
        "Relation_hits_at_3_{}".format(eval_name): hits_at_k(
            rel_ranks_filtered, 3),
        "Relation_hits_at_10_{}".format(eval_name): hits_at_k(
            rel_ranks_filtered, 10)
    }

    logger.info("Entity-Prediction Results:")
    logger.info(msg_validation_mrr_raw.format(
        values_dict["Entity_MR_{}".format(eval_name)],
        values_dict["Entity_MRR_{}".format(eval_name)]))
    logger.info(msg_validation_mrr_filtered.format(
        values_dict["Entity_MR_(filtered)_{}".format(eval_name)],
        values_dict["Entity_MRR_(filtered)_{}".format(eval_name)],
        head_proc_time + tail_proc_time))
    logger.info(msg_validation_hits.format(
        values_dict["Entity_hits_at_1_{}".format(eval_name)],
        values_dict["Entity_hits_at_3_{}".format(eval_name)],
        values_dict["Entity_hits_at_10_{}".format(eval_name)]))

    logger.info(">>> Subset head-prediction:")
    logger.info(msg_validation_hits.format(
        hits_at_k(head_ranks_filtered, 1),
        hits_at_k(head_ranks_filtered, 3),
        hits_at_k(head_ranks_filtered, 10)))
    logger.info(">>> Subset tail-prediction:")
    logger.info(msg_validation_hits.format(
        hits_at_k(tail_ranks_filtered, 1),
        hits_at_k(tail_ranks_filtered, 3),
        hits_at_k(tail_ranks_filtered, 10)))

    logger.info("Relation-Prediction Results:")
    logger.info(msg_validation_mrr_raw.format(
        values_dict["Relation_MR_{}".format(eval_name)],
        values_dict["Relation_MRR_{}".format(eval_name)]))
    logger.info(msg_validation_mrr_filtered.format(
        values_dict["Relation_MR_(filtered)_{}".format(eval_name)],
        values_dict["Relation_MRR_(filtered)_{}".format(eval_name)],
        head_proc_time + tail_proc_time))
    logger.info(msg_validation_hits.format(
        values_dict["Relation_hits_at_1_{}".format(eval_name)],
        values_dict["Relation_hits_at_3_{}".format(eval_name)],
        values_dict["Relation_hits_at_10_{}".format(eval_name)]))

    return values_dict


def __transe_ranking_template(session, config, dataset_init_op, op, op_filtered,
                              name):
    """Calculates the score (raw/filtered) for one entry of the tupel."""
    ranks = []
    ranks_filtered = []
    session.run([dataset_init_op])

    t_start = time.time()

    idx = 0
    with tqdm(total=config["total_validation_batches"],
              desc="{}".format(name)) as pbar:
        while True:
            try:
                batch_ranks, batch_ranks_filtered = session.run(
                    [op, op_filtered])

                # print(batch_ranks)
                ranks += list(batch_ranks)
                ranks_filtered += list(batch_ranks_filtered)

                idx += config["proc_units"]
                pbar.update(config["proc_units"])

                if config["truncated_validation"] > 1:
                    if idx >= config["truncated_validation"]:
                        pbar.close()
                        break
            except tf.errors.OutOfRangeError as e:
                break

    return ranks, ranks_filtered, time.time() - t_start
