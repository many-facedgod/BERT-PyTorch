from BERT.model.TrainableBERT import TrainableBERT
from BERT.utils.BERTTrainingDataset import BERTTrainingDataset
from BERT.utils.BERTLoss import BERTLoss
from BERT.utils.BERTTrainer import BERTTrainer
import torch
import torch.optim as optim
import pickle

vocab_dict = pickle.load(open("../data/vocab_dict.pkl", "rb"))
inv_vocab_dict = pickle.load(open("../data/inv_vocab_dict.pkl", "rb"))

ds = BERTTrainingDataset(total_size=1000)

config = {}
config["hidden_size"] = 256
config["vocab_size"] = len(vocab_dict)
config["num_layers"] = 5
config["positional_learnt"] = True
config["n_heads"] = 2
config["attention_dropout_rate"] = 0.1
config["dropout_rate"] = 0.1
config["max_sentence_length"] = 128
config["bottleneck_size"] = 128
config["eps_value"] = 1e-12

model = TrainableBERT(config, vocab_dict, inv_vocab_dict).cuda()
criterion = BERTLoss()
trainer = BERTTrainer(ds)
trainer.train(model, criterion, optim.Adam(model.parameters()))