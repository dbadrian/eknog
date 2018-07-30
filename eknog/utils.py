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

import errno
import json
import logging
import logging.config
import os

import sys

# Logging
from collections.__init__ import OrderedDict


def setup_logging(
        path='logger.json',
        level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration"""
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)


# hacky python stuff
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def mkdir_p(path):
    """Creates folders 'mkdir -p' style"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def json_load(path):
    """Open file, load deserialize json and return contents"""
    with open(path, 'r') as f:
        return json.load(f)


def json_dump(path, data):
    """Open file, serialize data to json, return nothing"""
    with open(path, 'w') as f:
        json.dump(data, f)


def dict_to_config_str(config_dict):
    """Produces a version of a dict with keys (and some values) replaced with
    shorter string version to avoid problems with over long file names in
    tensorboard"""

    key_abrv = {
        "embedding_dimension": "ed",
        "loss_type": "lt",
        "initialize_uniform": "iu",
        "k_negative_samples": "ns",
        "distance_measure": "dm",
        "margin": "mrgn",
        "sample_negative_relations": "snr",
        "bias": "b",
        "feature_map_dimension": "fmd",
        "fix_conv_layers": "fcl",
        "fix_structure_embeddings": "fse",
        "fix_word_embeddings": "fwd",
        "pretrained_word_embeddings": "pwe",
        "max_words_per_sentence": "mwps",
        "vocab_dim": "vd",
        "filter_sizes": "fs",
        "dropout_keep_prob": "dkp",
        "description_mode": "dm",
        "l1_kernel_size": "l1ks",
        "l2_kernel_size": "l2ks",
        "entity_wd_type": "eWDt",
        "rel_wd_type": "rWDt",
        "type_wd": "tWD",
        "type_rel_wd": "trWD",
        "filt_top_t": "ftt",
        "filt_btm_t": "fbt",
        "emb_dropout": "edp"
    }

    val_abrv = {
        None: "X",
        False: "F",
        True: "T",
        "softplus": "sp"
    }

    entries = []

    for name, value in config_dict.items():
        key = key_abrv[name] if name in key_abrv else name
        if type(value) == str or type(value) == bool:
            value = val_abrv[value] if value in val_abrv else value
        if type(value) == list:
            value = "L" + "-".join([str(v) for v in value]) + "L"

        # Skip (='delete') variable_device, no ones cares and the escape symbol messes
        # with the generated file path
        if key == "variable_device":
            continue

        entries.append((key, value))

    return entries


def generate_config_str(config):
    """Takes a config of Trainer class, and produces a shorted string, which
    tensorboard can handle better.

    Models with many parameters will otherwise quickly create extremely long
    filenames, which can cause problems with your os."""
    entries = [
        ("ml", config["model_type"]),
        ("ds", config["dataset_type"]),
        ("lr", config["optimizer_settings"]["learning_rate"]),
        ("bs", config["train_batch_size"])
    ]

    entries += dict_to_config_str(config["model_configuration"])

    return "_".join(k + "=" + str(v) for k, v in OrderedDict(entries).items())
