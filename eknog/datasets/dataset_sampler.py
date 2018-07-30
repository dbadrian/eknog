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
import os
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, save_npz, load_npz

logger = logging.getLogger(__name__)


def collect_from_set(x, entry, target, pivot):
    if entry == 0:
        t = x[1:]
    elif entry == 1:
        t = (x[0], x[2])
    else:
        t = x[:2]

    if t == target:
        col = x[entry]
        return col + 1 if col < pivot else col
    else:
        return None


class DatasetSampler():
    """
    This helper class helps with creating tensorflow dataset iterators for train
    validation, and testing.

    It precomputes various tables and stores compressed, sparse matrices on disk
    to avoid the boot-up time in the future.
    """

    def __init__(self, dataset, train_batch_size=2048,
                 evaluation_batch_size=10):
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.evaluation_batch_size = evaluation_batch_size

        self.train_set = None
        self.head_eval_set = None
        self.validation_tail_set = None
        self.test_head_set = None
        self.test_tail_set = None

        # Load sparse filter matrices (for evaluation/validation), or creates
        # them if they dont exist yet.
        p_prestored = os.path.join(self.dataset.path, "__prestored")
        self.valid_head_mask, self.valid_rel_mask, self.valid_tail_mask = self.__load_transe_filter_masks(
            p_prestored, data_type="valid")
        self.test_head_mask, self.test_rel_mask, self.test_tail_mask = self.__load_transe_filter_masks(
            p_prestored, data_type="test")

        # Provides dynamic access to the filter masks (programmatic access)
        self.mask_map = {
            "transe":
                {
                    "valid": {
                        0: self.valid_head_mask,
                        1: self.valid_rel_mask,
                        2: self.valid_tail_mask
                    },
                    "test": {
                        0: self.test_head_mask,
                        1: self.test_rel_mask,
                        2: self.test_tail_mask
                    },
                }
        }

        if "oog" in self.dataset.data_type2array:
            self.oog_head_mask, self.oog_rel_mask, self.oog_tail_mask = self.__load_transe_filter_masks(
                p_prestored, data_type="oog")
            self.mask_map["transe"]["oog"] = {
                0: self.oog_head_mask,
                1: self.oog_rel_mask,
                2: self.oog_tail_mask
            }

    def __batched_train_logit_generator(self, batch_size=1):

        shuffle(self.hr2e)

        l = len(self.hr2e)
        for ndx in range(0, l, batch_size):

            if min(ndx + batch_size, l) - ndx != batch_size:
                break

            row = []
            col = []

            for row_idx, (header, entry_list) in enumerate(
                    self.hr2e[ndx:min(ndx + batch_size, l)]):
                row += [row_idx] * len(entry_list)
                col += entry_list

            data = np.ones_like(row, dtype=np.bool)
            logit_mask = csr_matrix((data, (row, col)), shape=(
            min(ndx + batch_size, l) - ndx, self.dataset.n_entities),
                                    dtype=np.bool)

            hr = [hr for hr, _ in self.hr2e[ndx:min(ndx + batch_size, l)]]
            yield hr, logit_mask.toarray()

    def create_train_iterator(self, activate_bce=False):
        """
        Creates batched tf.dataset iterator which return train triplets only.
        """
        if activate_bce:
            hr2e = {(h, r): [] for h, r, t in self.dataset.train_set}
            [hr2e[(h, r)].append(t) for h, r, t in self.dataset.train_set]
            self.hr2e = [(hr, tuple(e)) for hr, e in hr2e.items()]

            # print("BCE-Loss, # train: ", len(self.hr2e))

            self.train_set = tf.data.Dataset.from_generator(
                lambda: self.__batched_train_logit_generator(
                    batch_size=self.train_batch_size), (tf.int32, tf.bool), (
                (self.train_batch_size, 2),
                (self.train_batch_size, self.dataset.n_entities)))
        else:
            self.train_set = tf.data.Dataset.from_tensor_slices(
                list(self.dataset.train_set))
            self.train_set = self.train_set.cache().shuffle(buffer_size=150000)
            self.train_set = self.train_set.prefetch(self.train_batch_size * 50)
            self.train_set = self.train_set.batch(self.train_batch_size)

        return self.train_set.make_initializable_iterator()

    def create_pr_evaluation_iterators(self, data_type='valid'):

        head_eval_set = tf.data.Dataset.from_generator(
            lambda: self.__triplet_group_to_batch(0, data_type=data_type,
                                                  batch_size=self.evaluation_batch_size),
            (tf.int32, tf.bool, tf.bool), (
            (None, 2), (None, self.dataset.n_entities),
            (None, self.dataset.n_entities)))
        head_eval_iterator = head_eval_set.make_initializable_iterator()

        rel_eval_dataset = tf.data.Dataset.from_generator(
            lambda: self.__triplet_group_to_batch(1, data_type=data_type,
                                                  batch_size=self.evaluation_batch_size),
            (tf.int32, tf.bool, tf.bool), (
            (None, 2), (None, self.dataset.n_relations),
            (None, self.dataset.n_relations)))
        rel_eval_iterator = rel_eval_dataset.make_initializable_iterator()

        tail_eval_dataset = tf.data.Dataset.from_generator(
            lambda: self.__triplet_group_to_batch(2, data_type=data_type,
                                                  batch_size=self.evaluation_batch_size),
            (tf.int32, tf.bool, tf.bool), (
            (None, 2), (None, self.dataset.n_entities),
            (None, self.dataset.n_entities)))
        tail_eval_iterator = tail_eval_dataset.make_initializable_iterator()

        return head_eval_iterator, rel_eval_iterator, tail_eval_iterator

    def triplet_group_to_batch(self, entry, data_type='valid', batch_size=1):
        return self.__triplet_group_to_batch(entry, data_type=data_type,
                                             batch_size=batch_size)

    def __triplet_group_to_batch(self, entry, data_type='valid', batch_size=1):
        data = self.dataset.data_type2array[data_type]
        dim = self.dataset.n_relations if entry == 1 else self.dataset.n_entities
        group = self.__group_triplets_by_entry(data, entry)

        # make a fixed order
        group = list(group.items())
        truth_mask = self.__triplet_group_to_mask(group, dim)
        entry_set = [t[0] for t in group]

        # obtain combined sets form the other datasets (other potential truth
        # values we dont want to test for
        alt_dtype = self.dataset.valid_data_types[:]
        alt_dtype.remove(data_type)
        alt_data = self.dataset.data_type2array[alt_dtype[0]] + \
                   self.dataset.data_type2array[alt_dtype[1]]
        alt = self.__group_triplets_by_entry(alt_data, entry)

        # filter alts
        filtered_alts = []
        for ident in entry_set:
            if ident in alt:
                filtered_alts.append((ident, alt[ident]))
            else:
                filtered_alts.append((ident, []))
        filter_mask = self.__triplet_group_to_mask(filtered_alts, dim)

        l = len(group)
        for ndx in range(0, l, batch_size):
            partial_truth_mask = truth_mask[ndx:min(ndx + batch_size, l),
                                 :].toarray()
            partial_filter_mask = filter_mask[ndx:min(ndx + batch_size, l),
                                  :].toarray()
            yield entry_set[ndx:min(ndx + batch_size,
                                    l)], partial_truth_mask, partial_filter_mask

    def __triplet_group_to_mask(self, group, dim):
        row = []
        col = []
        for row_idx, (header, entry_list) in enumerate(group):
            for entry in entry_list:
                row.append(row_idx)
                col.append(entry)

        data = np.ones_like(row, dtype=np.bool)
        return csr_matrix((data, (row, col)), shape=(len(group), dim),
                          dtype=np.bool)

    def __group_triplets_by_entry(self, data, entry):
        # collect the common head/tail head/rel and rel/tail combinations

        # determine a general slice
        if entry == 0:
            s = slice(1, 3)
        elif entry == 1:
            s = slice(0, 3, 2)
        else:
            s = slice(0, 2)

        # the possible configurations
        group = {conf: [] for conf in {t[s] for t in data}}
        [group[t[s]].append(t[entry]) for t in data]

        return group

    def create_transe_evaluation_iterators(self, data_type):
        """
        Generate tf.dataset iterators for validation or testing dataset.
        :param data_type: "valid" or "test". Returns head/rel/tal iterator,
        where the either head/rel/tail of a test triplet is replaced by all
        alternatives.

        :type data_type: str
        :return: head_iterator, rel_iterator and tail_iterator
        :rtype: initializable tf.dataset.iterator
        """
        dim_entities = self.dataset.n_entities
        if data_type == "oog":
            dim_entities += self.dataset.n_oog_entities

        head_eval_set = tf.data.Dataset.from_generator(
            lambda: self.__batched_eval_set_generator(0, data_type,
                                                      batch_size=self.evaluation_batch_size),
            (tf.int32, tf.float32), ((None, 3), (None, dim_entities)))
        head_eval_iterator = head_eval_set.make_initializable_iterator()

        rel_eval_dataset = tf.data.Dataset.from_generator(
            lambda: self.__batched_eval_set_generator(1, data_type,
                                                      batch_size=self.evaluation_batch_size),
            (tf.int32, tf.float32),
            ((None, 3), (None, self.dataset.n_relations)))
        rel_eval_iterator = rel_eval_dataset.make_initializable_iterator()

        tail_eval_dataset = tf.data.Dataset.from_generator(
            lambda: self.__batched_eval_set_generator(2, data_type,
                                                      batch_size=self.evaluation_batch_size),
            (tf.int32, tf.float32), ((None, 3), (None, dim_entities)))
        tail_eval_iterator = tail_eval_dataset.make_initializable_iterator()

        return head_eval_iterator, rel_eval_iterator, tail_eval_iterator

    def __batched_eval_set_generator(self, entry, data_type="valid",
                                     batch_size=1):
        """
        Generates a batches for evaluation (validation/testing).
        :param entry: Int between 0-2, with 0:head, 1: rel, 2:tail
        :type entry: int
        :param data_type: "valid" or "test"
        :type data_type: str
        :param batch_size: Size of batch
        :type batch_size: int
        :return: batch of test triplets, batch of filter masks for each triplet
        :rtype: [batch_size, 3], [batch_size, n_entities/n_relation]
        """
        assert entry == 0 or entry == 1 or entry == 2, "Entry is not {0,1,2}."

        data = self.dataset.data_type2array[data_type]
        l = len(data)
        for ndx in range(0, l, batch_size):
            # TODO: This 99999 score manipulation is terribly silly. I should do this somewhere else.
            # TODO: Maybe just return the bool mask, and perform the transformation on the function (e.g., rank) level
            yield data[ndx:min(ndx + batch_size, l)], \
                  self.mask_map["transe"][data_type][entry][
                  ndx:min(ndx + batch_size, l), :].toarray()

    def __load_transe_filter_masks(self, path, data_type="valid"):
        """
        Helper function to conveniently load all the filter masks for head/rel/tail
        at once.
        """
        if os.path.isfile(
                os.path.join(path, "{}_head_mask.npz".format(data_type))):
            # TODO: Loggin for there and all following functions!
            logger.info("Loading filter masks for {} dataset".format(data_type))
            head_mask = load_npz(
                os.path.join(path, "{}_head_mask.npz".format(data_type)))
            rel_mask = load_npz(
                os.path.join(path, "{}_rel_mask.npz".format(data_type)))
            tail_mask = load_npz(
                os.path.join(path, "{}_tail_mask.npz".format(data_type)))
        else:
            head_mask, rel_mask, tail_mask = self.__precompute_transe_filter_masks(
                data_type=data_type)
            save_npz(os.path.join(path, "{}_head_mask.npz".format(data_type)),
                     head_mask)
            save_npz(os.path.join(path, "{}_rel_mask.npz".format(data_type)),
                     rel_mask)
            save_npz(os.path.join(path, "{}_tail_mask.npz".format(data_type)),
                     tail_mask)

        return head_mask, rel_mask, tail_mask

    def __precompute_transe_filter_masks(self, data_type="valid"):
        """Helper function to compute head, rel and tail filters for a given
        dataset type (valid/test)."""

        logger.info("Generating Filter Masks for '{}'-set".format(data_type))

        union_set = frozenset(self.dataset.train_set) | frozenset(
            self.dataset.test_set) | frozenset(self.dataset.valid_set)

        head_filter_mask = self.__generate_transe_sparse_filter_mask(0,
                                                                     data_type,
                                                                     union_set)
        rel_filter_mask = self.__generate_transe_sparse_filter_mask(1,
                                                                    data_type,
                                                                    union_set)
        tail_filter_mask = self.__generate_transe_sparse_filter_mask(2,
                                                                     data_type,
                                                                     union_set)

        return head_filter_mask, rel_filter_mask, tail_filter_mask

    def __generate_transe_sparse_filter_mask(self, entry, data_type, union_set):
        """
        Generates a sparse matrix of false/true values for all possible alternatives
        of the test sample, depending on whether those alternatives are contained
        in the train/test/validation set (c.f., Bordes 2013, TransE Paper,
        filtered evaluation).

        True indicates that (under closed world assumption) the alternative
        triplet is contained in one of the train/test/valid sets.

        Each row relates one sample in the test or validation set, with rows ordered
        like how they are ordered in the dataset.

        Columns correspond to the index an entity maps to.

        :param entry: Int between 0-2, with 0:head, 1: rel, 2:tail
        :type entry: int
        :param data_type: Normally, "valid" or "test"
        :type data_type: str
        :return: Sparse matrix (compressed sparse row) of True/False
        :rtype:
        """
        assert type(self.dataset.data_type2array[
                        data_type]) == list, "Dataset is not a list. Make sure the dataset is a list (not a set) or the order will change."
        # dim = self.dataset.n_relations if entry == 1 else self.dataset.n_entities
        # if data_type == "oog" and entry != 1:
        #     dim += self.dataset.n_oog_entities
        #     self.dataset.union_set = self.dataset.union_set | set(self.dataset.oog_set)

        logger.debug("Generating {}-{}-Filter Masks".format(entry, data_type))

        dim = self.dataset.n_relations if entry == 1 else self.dataset.n_entities

        # generate lookup table
        lookup = self.__group_triplets_by_entry(union_set, entry)
        if entry == 0:
            s = slice(1, 3)
        elif entry == 1:
            s = slice(0, 3, 2)
        else:
            s = slice(0, 2)

        # Turn into a sparse matrix
        row = []
        col = []
        for row_idx, triplet in enumerate(
                self.dataset.data_type2array[data_type]):
            cols = lookup[triplet[s]]
            col += cols
            row += [row_idx] * len(cols)

        data = np.ones_like(row, dtype=np.bool)
        return csr_matrix((data, (row, col)), shape=(
        self.dataset.number_of_entries[data_type], dim), dtype=np.bool)
