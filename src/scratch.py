from BERT.model.TrainableBERT import TrainableBERT
from BERT.utils.BERTTrainingDataset import BERTTrainingDataset
from BERT.utils.BERTLoss import BERTLoss
from BERT.utils.BERTTrainer import BERTTrainer
from BERT.utils.BERTTokenizer import load_vocab
from tqdm import tqdm
import torch
from BERT.utils.BERTLoader import load_from_file
import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    vocab_dict, inv_vocab_dict = load_vocab("../data/uncased_vocab.txt")
    ds = BERTTrainingDataset(total_size=100000, batch_size=32)
    config, weights = load_from_file('/home/tanmaya/Work/CMU/11-747/MT-DNN-SOTA-Self/pretrained/weights/bert-base-uncased.tar.gz')
    model = TrainableBERT(config, vocab_dict, inv_vocab_dict, weights).cuda()
    criterion = BERTLoss()
    model.eval()
    losses = 0.0
    diffs = []
    count = 0
    lens = 0
    for batch in tqdm(ds):
        count = 0
        lens = 0
        for sentence in batch['input']['sentences']:
            for j in sentence:
                if inv_vocab_dict[j.item()] == '[MASK]':
                    count += 1
            #print([inv_vocab_dict[i.item()] for i in sentence])
        #batch['input']['sentences'][batch["tags"]["lm_labels"][0], batch["tags"]["lm_labels"][1]] = batch["tags"]["lm_labels"][2]
        lens += len(batch["tags"]["lm_labels"][1])
        diffs.append(count - lens)
        #print(len(batch["tags"]["lm_labels"][1]))
        #print(len(batch["tags"]["lm_labels"][0]))
        #print(batch["tags"]["lm_labels"][1].shape)
        #print(count)
        #print("###################")
        """for sentence in batch['input']['sentences']:
            print([inv_vocab_dict[i.item()] for i in sentence])"""
        #break
    print(count)
    print(count - lens)
    plt.plot(diffs)
    plt.show()
    """with torch.no_grad():
        for batch in tqdm(ds):
            predictions = model(batch["input"])
            loss = criterion(predictions, batch["tags"])
            losses += loss.item()
    print(losses / len(ds))"""

    #trainer = BERTTrainer(ds)
    #trainer.train(model, criterion, optim.Adam(model.parameters()), save_every=-6500)


if __name__ == "__main__":
    main()
