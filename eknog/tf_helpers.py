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
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import timeline, device_lib


def variable_on_device(name, shape, dtype, initializer, trainable,
                       device='/cpu:0',
                       weight_decay=None, wd_func=tf.nn.l2_loss):
    """Helper to create a variable on specified device. Defaults to CPU.

      If weight decay is specified, a L2 weight decay is added. HOWEVER, depending
      on how you calculate the L2 loss, this might not be exactly how you want it.
    """
    with tf.device(device):
        v = tf.get_variable(name=name, shape=shape, dtype=dtype,
                            initializer=initializer, trainable=trainable)

        if weight_decay is not None:
            wd = tf.multiply(wd_func(v), weight_decay,
                             name=name + '_weight_decay')

            tf.add_to_collection('weight_decay', wd)

        return v

def write_timeline(run_metadata, file_path):
    """Convenience function to generate and save a chrome trace format timeline."""
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(file_path, 'w') as f:
        f.write(ctf)


def maching_variable_restore(session, file_name):
    """
    Restores all variables from the checkpoint which are also present in the current graph.
    Derived models can be initialized from a base models, which contain a subset of their variables.
    """
    chkp_var_shape_map = pywrap_tensorflow.NewCheckpointReader(
        file_name).get_variable_to_shape_map()
    variables_in_graph = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    names_in_graph = {var.name.split(":")[0]: var for var in variables_in_graph}

    mapping = {name: names_in_graph[name] for name, _ in
               chkp_var_shape_map.items() if name in names_in_graph}

    saver = tf.train.Saver(var_list=mapping)
    saver.restore(session, file_name)

    return mapping
