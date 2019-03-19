from BERT.model.BERT import BERT

config = {}
config["hidden_size"] = 10
config["vocab_size"] = 20
config["num_layers"] = 3
config["positional_learnt"] = True
config["n_heads"] = 2
config["attention_dropout_rate"] = 0.1
config["dropout_rate"] = 0.1
config["max_sentence_length"] = 10
config["intermediate_size"] = 10
config["eps_value"] = 1e-12

x = BERT(config, {}, {})
print(x)