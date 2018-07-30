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

import tensorflow as tf


class SummaryGroup():
    """
    Utility class to group summaries together and update them together.

    Additional (simple) non-TF values can be supplied at the time of writing
    summaries.
    """

    def __init__(self, fn_folder, graph=None, writer=None):
        self.list_of_summaries = []

        if writer is None:
            self.writer = tf.summary.FileWriter(fn_folder, graph)
        else:
            self.writer = writer

    def add_graph(self, graph):
        self.writer.add_graph(graph)

    def add_summary(self, summary):
        if type(summary) == list:
            self.list_of_summaries += summary
        else:
            self.list_of_summaries.append(summary)

    def merge_summaries(self):
        return tf.summary.merge(self.list_of_summaries)

    def write(self, summaries=None, global_step=None, values_dict=None):
        if summaries:
            # Run any regular tf-summaries first, and write to file
            self.writer.add_summary(summaries, global_step)

        # If called with a list of values, process them as well
        if values_dict:
            for tag, value in values_dict.items():
                summary = tf.Summary.Value(tag=tag, simple_value=value)
                summary = tf.Summary(value=[summary])
                self.writer.add_summary(summary, global_step)

        # ensure all data is written to the disk properly
        self.writer.flush()
