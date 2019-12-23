# BERT in PyTorch
A clean implementation of Bidirectional Encoder Representations from Transformers proposed by Devlin et. al. in the paper [
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

## Requirements
- Python 3
- PyTorch >= 1.1
- tqdm
- Numpy

## Training your own BERT

To train your own BERT you need to tokenize your dataset into an array of array of integers, with each array of integers representing one sentence, and consecutive array of arrays should be consecutive sentences in the corpus. You may find the BERTTokenizer (copied from [here](https://github.com/google-research/bert)) useful. You should also have a txt file listing the vocabulary, including the `[SEP]`, `[MASK]`, `[PAD]` and `[CLS]` tokens. Each vocabulary item should be listed in a new line. Examples are provided in the data directory. You should then like to initialize the BERT dataset and the BERT trainer and run the train method as shown in [examples.py](./src/examples.py).

## Loading the Huggingface weights

The pre-trained TensorFlow weights have been ported to PyTorch by [huggingface](https://github.com/huggingface/transformers). If you want to use the pre-trained weights, you can use the function provided in [HuggingfaceUtils.py](./src/BERT/utils/HuggingfaceUtils.py). An example usage is provided in [examples.py](./src/examples.py).

