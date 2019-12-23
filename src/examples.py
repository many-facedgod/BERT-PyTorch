import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm

from BERT.model.BERT import BERT
from BERT.model.TrainableBERT import TrainableBERT
from BERT.utils.BERTLoss import BERTLoss
from BERT.utils.BERTTokenizer import load_vocab
from BERT.utils.BERTTrainer import BERTTrainer
from BERT.utils.BERTTrainingDataset import BERTTrainingDataset
from BERT.utils.HuggingfaceUtils import load_huggingface_pretrained_bert


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ds = BERTTrainingDataset(total_size=1000, batch_size=5, device=device)

    config = {
        "hidden_size": 256,  # Hidden size for the network
        "vocab_size": 30522,  # The size of the vocabulary including [MASK], [PAD], [CLS], and [SEP]
        "num_layers": 3,  # Number of transformer blocks
        "positional_learnt": False,  # Whether to learn positional embeddings or use sinusoids
        "n_heads": 2,  # Number of attention heads
        "attention_dropout_rate": 0.,  # Dropout probability for attention
        "dropout_rate": 0.,  # Dropout probability for feed-forward layers
        "max_sentence_length": 100,  # Maximum sentence length
        "bottleneck_size": 256,  # Bottleneck size for the residual links
        "eps_value": 1e-12  # Epsilon for layer normalization
    }
    model = TrainableBERT(config).to(device)
    criterion = BERTLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    trainer = BERTTrainer(ds, run_desc="toy_run", log=True)
    trainer.train(model, criterion, optimizer, iters=5, save_every=100)

    del model
    del optimizer

    print("Loading pre-trained BERT...")
    config, state_dict = load_huggingface_pretrained_bert("../pretrained/bert-base-uncased.tar.gz")
    model = TrainableBERT(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("Evaluating pre-trained BERT...")
    with torch.no_grad():
        losses = []
        for batch in tqdm(ds):
            losses.append(criterion(model(batch["input"]), batch["tags"]))
        print(f"Loss for pre-trained BERT: {np.mean(losses)}")

    del model
    del state_dict
    del ds

    # You should download this from the HuggingFace repository
    config, state_dict = load_huggingface_pretrained_bert("../pretrained/bert-base-uncased.tar.gz", False)
    model = BERT(config)
    model.load_state_dict(state_dict)
    model = model.to(device)

    vocab, _ = load_vocab("../data/uncased_vocab.txt")
    sentence = np.load("../data/dataset_toy.npy")[0]
    print("Encoding the first sentence in the dataset")

    # Adding CLS and SEP token
    input_ = np.full((1, len(sentence) + 2), vocab["[PAD]"], dtype=np.int64)
    input_[0, 1:len(sentence) + 1] = sentence
    input_[0, 0] = vocab['[CLS]']
    input_[0, len(sentence) + 1] = vocab['[SEP]']
    input_ = torch.tensor(input_).to(device)
    sentence_type = torch.ones_like(input_)
    sentence_mask = torch.ones_like(input_).float()
    batch = {"sentences": input_, "sentence_type": sentence_type, "sentence_mask": sentence_mask}
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=10)
    with torch.no_grad():
        print(model(batch)[1][0].cpu().numpy())


if __name__ == "__main__":
    main()
