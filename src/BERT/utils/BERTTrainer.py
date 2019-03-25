import torch
from os.path import realpath, join, isdir, dirname
from os import mkdir
import sys
from tqdm import tqdm
import time
import numpy as np

curr_dir = dirname(realpath(__file__))


class BERTTrainer:

    def __init__(self, dataset, exp_path=join(curr_dir, "../../../experiments"), run_desc="", log=True):
        """
        :param dataset: The dataset to be used
        :param exp_path: The path to the experiments directory
        :param run_desc: The description of this run
        :param log: Whether to log or not
        """
        self.dataset = dataset
        self.exp_path = exp_path
        if not run_desc:
            self.run_desc = " ".join(sys.argv[1:])
        else:
            self.run_desc = run_desc
        if not isdir(exp_path):
            mkdir(exp_path)
        self.run_id = str(int(time.time()))
        mkdir(join(exp_path, self.run_id))
        mkdir(join(exp_path, self.run_id, "models"))
        self.log_file = open(join(exp_path, self.run_id, "log.txt"), "w") if log else None
        descriptions_file = open(join(exp_path, "descriptions.txt"), "a")
        descriptions_file.write("{}: {}\n".format(self.run_id, self.run_desc))
        descriptions_file.close()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.log("--------------------------------------------------------")

    def log(self, x):
        """Logs a string to the log file and stdout
        """
        tqdm.write(x)
        if self.log_file is not None:
            self.log_file.write(str(x) + "\n")
            self.log_file.flush()

    @staticmethod
    def save_state(exp_path, run_id, tags, model=None, optimizer=None):
        """Saves the model and optimizer states

        Args:
            exp_path (str): The experiments path
            run_id (str): The current run ID
            tags (tuple): Tags for the model to be saved
            model (nn.Module, optional): Defaults to None. The model to be saved
            optimizer (optim.Optimizer, optional): Defaults to None. The optimizer to be saved
        """

        name = ".".join([str(i) for i in tags]) + ".pt"
        state = {}
        if model:
            state["model_state"] = model.state_dict()
        if optimizer:
            state["optim_state"] = optimizer.state_dict()
        path = join(exp_path, run_id, "models", name)
        torch.save(state, path)

    @staticmethod
    def load_state(exp_path, run_id, tags, device, model=None, optimizer=None, strict=False):
        """Load a saved model

        Args:
            exp_path (str): The experiment path
            run_id (int): The run ID of the model to be loaded
            tags (tuple): Tags for the model to be loaded
            device (torch.device): The device to which the model is to be loaded
            model (nn.Module, optional): Defaults to None. The target model for loadin the state
            optimizer (optim.Optimizer, optional): Defaults to None. The target optimizer for loading the state
            strict (bool, optional): Defaults to False. Strict or lenient loading
        """

        name = ".".join([str(i) for i in tags]) + ".pt"
        run_id = str(run_id)
        path = join(exp_path, run_id, "models", name)
        state = torch.load(path, map_location=device)
        if model is not None:
            model.load_state_dict(state["model_state"], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(state["optim_state"])

    def train(self, model, criterion, optimizer, iters=20, save_every=5, preload=None, load_optimizer=False):
        """Trains a model

        Args:
            model (nn.Module): The model to be trained
            criterion (nn.Module): The loss function
            optimizer (optim.Optimizer): The optimizer to use
            iters (int): The number of iterations
            save_every (int, optional): Defaults to 5. Save every k iterations
            preload (tuple, optional): Defaults to None. tuple of (run_id, tags) for preloading
            load_optimizer (bool, optional): Defaults to False. If preload is not None, whether to load the optimizer or not
        """

        if preload is not None:
            if load_optimizer:
                self.log("Loading model and optimizer {} from run_id {}...".format(str(preload[1]), preload[0]))
                BERTTrainer.load_state(self.exp_path, preload[0], preload[1], self.device, model, optimizer)
            else:
                self.log("Loading model {} from run_id {}...".format(str(preload[1]), preload[0]))
                BERTTrainer.load_state(self.exp_path, preload[0], preload[1], self.device, model)
            self.log("Loaded.")
            self.log("--------------------------------------------------------")
        for i in range(1, iters + 1):
            losses = []
            self.log("Iteration {}/{}:".format(i, iters))
            bar = tqdm(self.dataset, desc="Current training loss: NaN", file=sys.stdout)
            for j, batch in enumerate(bar):
                predictions = model(batch["input"])
                loss = criterion(predictions, batch["tags"])
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_description("Current training loss: {}".format(loss.item()))
                if save_every < 0 and not j % -save_every:
                    self.log("Saving model...")
                    BERTTrainer.save_state(self.exp_path, self.run_id, ("checkpoint", i, j), model, optimizer)
                    self.log("Saved model {}".format(str(("checkpoint", i, j))))
                    self.log("--------------------------------------------------------")
            self.log("Mean loss for the iteration: {}".format(np.mean(losses)))
            self.log("--------------------------------------------------------")
            if save_every > 0 and not i % save_every:
                self.log("Saving model...")
                BERTTrainer.save_state(self.exp_path, self.run_id, ("checkpoint", i), model, optimizer)
                self.log("Saved model {}".format(str(("checkpoint", i))))
                self.log("--------------------------------------------------------")
