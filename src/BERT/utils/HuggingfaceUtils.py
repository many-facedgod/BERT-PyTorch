import json
import tarfile

import torch


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


def _make_dict(weights_dict, config, trainable):
    weights_dict["embeddings.token_embedding.weight"] = weights_dict.pop("bert.embeddings.word_embeddings.weight")
    weights_dict["embeddings.type_embedding.weight"] = weights_dict.pop("bert.embeddings.token_type_embeddings.weight")
    weights_dict["embeddings.positional_embedding.weight"] = weights_dict.pop("bert.embeddings.position_"
                                                                              "embeddings.weight")
    weights_dict["embeddings.layer_norm.layer_norm.weight"] = weights_dict.pop("bert.embeddings.LayerNorm.gamma")
    weights_dict["embeddings.layer_norm.layer_norm.bias"] = weights_dict.pop("bert.embeddings.LayerNorm.beta")
    weights_dict["pooler.weight"] = weights_dict.pop("bert.pooler.dense.weight")
    weights_dict["pooler.bias"] = weights_dict.pop("bert.pooler.dense.bias")
    for i in range(config["num_layers"]):
        block = f"layers.{i}."
        weights_dict[f"{block}attention.query_map.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                              f"attention.self.query.weight")
        weights_dict[f"{block}attention.query_map.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                            f"attention.self.query.bias")
        weights_dict[f"{block}attention.key_map.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                            f"attention.self.key.weight")
        weights_dict[f"{block}attention.key_map.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                          f"attention.self.key.bias")
        weights_dict[f"{block}attention.value_map.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                              f"attention.self.value.weight")
        weights_dict[f"{block}attention.value_map.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                            f"attention.self.value.bias")
        weights_dict[f"{block}attention.ff.linear.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                              f"attention.output.dense.weight")
        weights_dict[f"{block}attention.ff.linear.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                            f"attention.output.dense.bias")
        weights_dict[f"{block}attention.ff.layer_norm.layer_norm.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                                             f"attention.output."
                                                                                             f"LayerNorm.gamma")
        weights_dict[f"{block}attention.ff.layer_norm.layer_norm.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                                           f"attention.output."
                                                                                           f"LayerNorm.beta")
        weights_dict[f"{block}bottleneck.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                     f"intermediate.dense.weight")
        weights_dict[f"{block}bottleneck.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}.intermediate.dense.bias")
        weights_dict[f"{block}output.linear.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}.output.dense.weight")
        weights_dict[f"{block}output.linear.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}.output.dense.bias")
        weights_dict[f"{block}output.layer_norm.layer_norm.weight"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                                       f"output.LayerNorm.gamma")
        weights_dict[f"{block}output.layer_norm.layer_norm.bias"] = weights_dict.pop(f"bert.encoder.layer.{i}."
                                                                                     f"output.LayerNorm.beta")

    if trainable:
        for key in list(weights_dict.keys()):
            weights_dict[f"bert.{key}"] = weights_dict.pop(key)
        weights_dict["lm_l1.weight"] = weights_dict.pop("bert.cls.predictions.transform.dense.weight")
        weights_dict["lm_l1.bias"] = weights_dict.pop("bert.cls.predictions.transform.dense.bias")
        weights_dict["lm_ln.layer_norm.weight"] = weights_dict.pop("bert.cls.predictions.transform.LayerNorm.gamma")
        weights_dict["lm_ln.layer_norm.bias"] = weights_dict.pop("bert.cls.predictions.transform.LayerNorm.beta")
        weights_dict["lm_l2.weight"] = weights_dict.pop("bert.cls.predictions.decoder.weight")
        weights_dict["lm_l2.bias"] = weights_dict.pop("bert.cls.predictions.bias")
        weights_dict["nsp_l1.weight"] = weights_dict.pop("bert.cls.seq_relationship.weight")
        weights_dict["nsp_l1.bias"] = weights_dict.pop("bert.cls.seq_relationship.bias")
    else:
        to_pop = ["cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
                  "cls.predictions.transform.LayerNorm.gamma", "cls.predictions.transform.LayerNorm.beta",
                  "cls.predictions.decoder.weight", "cls.predictions.bias", "cls.seq_relationship.weight",
                  "cls.seq_relationship.bias"]
        for key in to_pop:
            del weights_dict[key]


def load_huggingface_pretrained_bert(tar_path, trainable=True):
    """Loads the BERT models converted to PyTorch by HuggingFace: https://github.com/huggingface/transformers"""
    tf = tarfile.open(tar_path)
    config_file = tf.getmember("./bert_config.json")
    config = _make_config(json.loads(tf.extractfile(config_file).read()))
    weights_file = None
    for member in tf.getmembers():
        if member != config_file:
            weights_file = member
    assert weights_file is not None, "No weights file available"
    weights_dict = torch.load(tf.extractfile(weights_file))
    _make_dict(weights_dict, config, trainable)
    return config, weights_dict
