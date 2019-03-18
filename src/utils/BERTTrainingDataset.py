import numpy as np
import pickle
import torch
from os.path import join, realpath, dirname

curr_dir = dirname(realpath(__file__))


class BERTTrainingDataset:

    def __init__(self, data_path=join(curr_dir, "../data"), total_size=15000000, batch_size=8, noise_prob=0.15,
                 wrong_sent_prob=0.5, device=torch.device("cuda")):
        """
        :param data_path: Path to where the dataset and the vocab files are
        :param total_size: What portion of the dataset to use
        :param batch_size: The batch size
        :param noise_prob: Noise probability for the denoising language model
        :param wrong_sent_prob: Probability for the next sentence to be a random one
        :param device: The device to which the tensors are to be cast
        """
        self.dataset = np.load(join(data_path, "dataset_toy.npy"))[:total_size]
        self.vocab = pickle.load(open(join(data_path, "vocab_dict.pkl"), "rb"))
        self.inv_vocab = np.load(open(join(data_path, "inv_vocab_dict.pkl"), "rb"))
        self.batch_size = batch_size
        self.noise_prob = noise_prob
        self.wrong_sent_prob = wrong_sent_prob
        self.device = device

    def __len__(self):
        return (len(self.dataset) - 1) // self.batch_size + 1

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        n_batches = len(self)
        for i in range(n_batches):
            yield self._make_batch(indices[i * self.batch_size: (i + 1) * self.batch_size])

    def _mask(self, data_chunk):
        ind_x, ind_y, label = [], [], []
        for i in range(len(data_chunk)):
            noise_predicate = np.random.random(size=len(data_chunk[i])) < self.noise_prob
            # Add [MASK] to 80%
            mask_predicate = np.logical_and(np.random.random(size=len(data_chunk[i])) < 0.8, noise_predicate)
            # A random word for 20%
            random_predicate = np.logical_and(np.random.random(size=len(data_chunk[i])) < 0.5, noise_predicate)
            random_predicate = np.logical_and(random_predicate, np.logical_not(mask_predicate))

            indices = np.where(noise_predicate)[0]
            ind_x.append(np.array([i] * len(indices), dtype=np.int64))
            ind_y.append(indices)
            label.append(data_chunk[i][indices])
            data_chunk[i][np.where(mask_predicate)[0]] = self.vocab["[MASK]"]
            data_chunk[i][np.where(random_predicate)[0]] = np.random.randint(len(self.vocab),
                                                                             size=random_predicate.sum())
        return data_chunk, np.concatenate(ind_x).astype(np.int64), np.concatenate(ind_y).astype(
            np.int64), np.concatenate(label).astype(np.int64)

    def _make_batch(self, indices):
        batch_size = len(indices)
        t1_labels = (np.random.random(size=batch_size) > self.wrong_sent_prob).astype(np.int64)
        first_sents, ind_x, ind_y, label = self._mask(self.dataset[indices])
        first_lengths = np.array([len(i) for i in first_sents])
        second_sents = self.dataset[np.random.randint(len(self.dataset), size=batch_size)]
        second_sents[np.where(t1_labels == 1)[0]] = self.dataset[(indices + 1) % len(self.dataset)][
            np.where(t1_labels == 1)[0]]
        second_sents, ind2_x, ind2_y, label2 = self._mask(second_sents)
        second_lengths = np.array([len(i) for i in second_sents])
        for i in range(batch_size):
            ind2_y[np.where(ind2_x == i)] += first_lengths[i] + 2
        ind_y += 1
        final_inds = np.concatenate([ind_x, ind2_x]), np.concatenate([ind_y, ind2_y]), np.concatenate([label, label2])
        t2_labels = tuple(torch.LongTensor(i).to(self.device) for i in final_inds)
        t1_labels = torch.LongTensor(t1_labels)
        max_len = max(first_lengths + second_lengths + 3)
        chunk = np.full((batch_size, max_len), self.vocab["[PAD]"], dtype=np.int64)
        chunk[:, 0] = self.vocab["[CLS]"]

        """sent_tag is to distinguish between the first and the second sentence: 0 is PAD, 1 is the first sentence, 
        2 is the second"""
        sent_tag = np.zeros((batch_size, max_len), dtype=np.int64)

        for i in range(batch_size):
            chunk[i, 1:first_lengths[i] + 1] = first_sents[i]
            chunk[i, first_lengths[i] + 1] = self.vocab["[SEP]"]
            sent_tag[i, :first_lengths[i] + 2] = 1
            chunk[i, first_lengths[i] + 2: first_lengths[i] + 2 + second_lengths[i]] = second_sents[i]
            sent_tag[i, first_lengths[i] + 2: first_lengths[i] + 3 + second_lengths[i]] = 2
            chunk[i, first_lengths[i] + 2 + second_lengths[i]] = self.vocab["[SEP]"]
        mask = np.zeros(chunk.shape)
        mask[np.where(chunk != self.vocab["[PAD]"])] = 1
        batch = {"input": {}, "tags": {}}
        batch["input"]["sentences"] = torch.LongTensor(chunk).to(self.device)
        batch["input"]["sentence_mask"] = torch.FloatTensor(mask).to(self.device)
        batch["input"]["sentence_type"] = torch.LongTensor(sent_tag).to(self.device)
        batch["tags"]["lm_labels"] = t2_labels
        batch["tags"]["nsp_labels"] = t1_labels
        return batch
