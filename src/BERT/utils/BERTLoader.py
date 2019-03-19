import torch
import tarfile
import json


def _make_config(loaded_config):
    config = {}
    config["hidden_size"] = loaded_config["hidden_size"]
    config["vocab_size"] = loaded_config["vocab_size"]
    config["num_layers"] = loaded_config["num_hidden_layers"]
    config["positional_learnt"] = True
    config["n_heads"] = loaded_config["num_attention_heads"]
    config["attention_dropout_rate"] = loaded_config["attention_probs_dropout_prob"]
    config["dropout_rate"] = loaded_config["hidden_dropout_prob"]
    config["max_sentence_length"] = loaded_config["max_position_embeddings"]
    config["bottleneck_size"] = loaded_config["intermediate_size"]
    config["eps_value"] = 1e-12
    return config


def load_from_file(bert_path):
    tf = tarfile.open(bert_path)
    config_file = tf.getmember("./bert_config.json")
    config = _make_config(json.loads(config_file.read()))
    weights_file = None
    for member in tf.getmembers():
        if member != config_file:
            weights_file = member
    assert weights_file is not None, "No weights file available"
    weights_dict = torch.load(weights_file)
    return config, weights_dict
