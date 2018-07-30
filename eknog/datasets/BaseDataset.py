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

import os
from abc import ABC

from ordered_set import OrderedSet

import eknog.utils as common


class BaseDataset(ABC):

    def __init__(self, dataset, file_name_pattern='%s.txt'):
        self.path = os.path.join(".", "data", dataset)
        self.__class__.__name__ = dataset

        # What files can/should be loaded. Potentially you can add further
        # for example, for out-of-graph/zeroshot tests.
        self.data_types = ["train", "test", "valid"]

        # Map for entity/rel identifiers to unique ids (int)
        self.entity2idx = {}
        self.rel2idx = {}

        # And the inverse mapping to above
        self.idx2entity = {}
        self.idx2rel = {}

        # # If the dataset contains a mapping from entity/idx to a human-readable-name
        # self.idx2name = {}

        # Not all datasets name files the same... no need to rename files if
        # they follow a pattern
        self.file_name_pattern = file_name_pattern

        self._load_base()

    def _load_base(self):
        """
        Loads the base data splits (train, test, valid) and initializes the mappings
        :return:
        :rtype:
        """

        # Check if pre-computed "tables" exist for faster loading
        fn_prestored = os.path.join(self.path, '__prestored')
        if os.path.isdir(fn_prestored):
            try:
                self.entity2idx = common.json_load(
                    os.path.join(fn_prestored, 'entity2idx.json'))
                self.rel2idx = common.json_load(
                    os.path.join(fn_prestored, 'rel2idx.json'))
                self.train_set = [tuple(l) for l in common.json_load(
                    os.path.join(fn_prestored, 'train_set.json'))]
                self.test_set = [tuple(l) for l in common.json_load(
                    os.path.join(fn_prestored, 'test_set.json'))]
                self.valid_set = [tuple(l) for l in common.json_load(
                    os.path.join(fn_prestored, 'valid_set.json'))]
            except FileExistsError as e:
                print(e)
        else:
            # load each data_type in order

            data = {
                "train": list(self._load_data_file("train")),
                "valid": list(self._load_data_file("valid")),
                "test": list(self._load_data_file("test")),
            }

            # Needs to be done over all datasets, as there are some defective
            # datasets like WN18RR or Yago3-10
            self._generate_unique_ids(
                data["train"][0] + data["valid"][0] + data["test"][0],
                data["train"][1] + data["valid"][1] + data["test"][1],
                data["train"][2] + data["valid"][2] + data["test"][2])

            for data_type in ["train", "test", "valid"]:
                heads, rels, tails = data[data_type]

                if data_type == "train":
                    self.train_set, self.train_oog = self._convert_names_to_ids(
                        heads, rels,
                        tails)
                    if self.train_oog:
                        print(self.train_oog)
                elif data_type == "test":
                    self.test_set, self.test_oog = self._convert_names_to_ids(
                        heads, rels,
                        tails)
                    if self.test_oog:
                        print(self.test_oog)
                elif data_type == "valid":
                    self.valid_set, self.valid_oog = self._convert_names_to_ids(
                        heads, rels,
                        tails)
                    if self.valid_oog:
                        print(self.valid_oog)

            # print("If the list are not empty, something is wrong with the data:", train_oog, valid_oog, test_oog)

            # Create folder and dump generated files to preloading
            common.mkdir_p(fn_prestored)
            common.json_dump(os.path.join(fn_prestored, 'entity2idx.json'),
                             self.entity2idx)
            common.json_dump(os.path.join(fn_prestored, 'rel2idx.json'),
                             self.rel2idx)
            common.json_dump(os.path.join(fn_prestored, 'train_set.json'),
                             self.train_set)
            common.json_dump(os.path.join(fn_prestored, 'test_set.json'),
                             self.test_set)
            common.json_dump(os.path.join(fn_prestored, 'valid_set.json'),
                             self.valid_set)

        # For easier access and checking if other data types are added
        self.data_type2array = {"train": self.train_set,
                                "test": self.test_set,
                                "valid": self.valid_set}

        # Set some useful variables
        self.n_entities = len(self.entity2idx)
        self.n_relations = len(self.rel2idx)
        self.number_of_entries = {"train": len(self.train_set),
                                  "test": len(self.test_set),
                                  "valid": len(self.valid_set)}

    def _load_data_file(self, data_type):
        """ Helper function to load the txt file specified by data_type and
        return it as a list of heads, relations and tails.

        :param data_type: "train", "test', or "valid"
        :type data_type: str
        :return: list for heads, relations and one for tails
        :rtype: list of strs
        """
        if data_type in self.data_types:
            with open(os.path.join(self.path,
                                   self.file_name_pattern % data_type)) as f:
                return zip(*(line.strip().split('\t') for line in f))
        else:
            raise ValueError("Chosen data type is not valid!")

    def _generate_unique_ids(self, heads, rels, tails):
        """ Takes list of heads, rels and tails, determines unique entities and
        relations and assigns unique ids. Results are stored in self.entity2idx
        and self.rel2idx.
        """
        self.heads = OrderedSet(heads)
        self.tails = OrderedSet(tails)
        self.relations = OrderedSet(rels)

        # construct sets of unique heads, tails and a set of shared entities
        unique_heads = self.heads - self.tails
        shared_entities = self.heads & self.tails
        unique_tails = self.tails - self.heads

        idx = 0
        for entity in unique_heads:
            self.entity2idx[entity] = idx
            self.idx2entity[idx] = entity
            idx += 1
        for entity in shared_entities:
            self.entity2idx[entity] = idx
            self.idx2entity[idx] = entity
            idx += 1
        for entity in unique_tails:
            self.entity2idx[entity] = idx
            self.idx2entity[idx] = entity
            idx += 1

        # Now add relations
        idx = 0
        for rel in self.relations:
            self.rel2idx[rel] = idx
            self.idx2rel[idx] = rel
            idx += 1

    def _convert_names_to_ids(self, heads, rels, tails):
        return list(zip(
            [self.entity2idx[head] for head in heads if
             head in self.entity2idx],
            [self.rel2idx[relation] for relation in rels if
             relation in self.rel2idx],
            [self.entity2idx[tail] for tail in tails if tail in self.entity2idx]
        )), list(zip(
            [head for head in heads if head not in self.entity2idx],
            [relation for relation in rels if relation not in self.rel2idx],
            [tail for tail in tails if tail not in self.entity2idx]
        ))
