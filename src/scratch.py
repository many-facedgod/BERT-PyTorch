from BERT.model.TrainableBERT import TrainableBERT
from BERT.utils.BERTTrainingDataset import BERTTrainingDataset
from BERT.utils.BERTLoss import BERTLoss
from BERT.utils.BERTTrainer import BERTTrainer
from BERT.utils.BERTTokenizer import load_vocab
import torch.optim as optim


def main():
    vocab_dict, inv_vocab_dict = load_vocab("../data/vocab.txt")
    ds = BERTTrainingDataset(total_size=1000000, batch_size=32)
    config = {}
    config["hidden_size"] = 768
    config["vocab_size"] = len(vocab_dict)
    config["num_layers"] = 12
    config["positional_learnt"] = True
    config["n_heads"] = 12
    config["attention_dropout_rate"] = 0.1
    config["dropout_rate"] = 0.1
    config["max_sentence_length"] = 512
    config["bottleneck_size"] = 3072
    config["eps_value"] = 1e-12
    model = TrainableBERT(config, vocab_dict, inv_vocab_dict).cuda()
    criterion = BERTLoss()
    trainer = BERTTrainer(ds)
    trainer.train(model, criterion, optim.Adam(model.parameters()), save_every=-6500)


if __name__ == "__main__":
    main()
