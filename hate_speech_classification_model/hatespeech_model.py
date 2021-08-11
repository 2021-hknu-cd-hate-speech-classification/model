import numpy as np
import torch
import re
import emoji
import soynlp
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


class HateSpeechClassifier(pl.LightningModule):
    def __init__(self, hyper_parameter: dict):
        super().__init__()

        ### 데이터셋 주소 ###
        self.TRAIN_SET_URL = "https://raw.githubusercontent.com/2021-hknu-cd-hate-speech-classification/dataset/main/split/train.csv"
        self.TEST_SET_URL = "https://raw.githubusercontent.com/2021-hknu-cd-hate-speech-classification/dataset/main/split/test.csv"

        ### 하이퍼파라미터 ###
        self.MAX_LENGTH = hyper_parameter["max_length"] if ("max_length" in hyper_parameter) else 150
        self.LEARNING_RATE = hyper_parameter["lr"] if ("lr" in hyper_parameter) else 5e-6
        self.EPOCHS = hyper_parameter["epochs"] if ("epochs" in hyper_parameter) else 5
        self.MODEL_NAME = hyper_parameter["model"] if ("model" in hyper_parameter) else "beomi/KcELECTRA-base"
        self.OPTIMIZER = hyper_parameter["optimizer"] if ("optimizer" in hyper_parameter) else "adamw"
        self.GAMMA = hyper_parameter["gamma"] if ("gamma" in hyper_parameter) else 0.5

        ### 사용할 모델 ###
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        if "kogpt2" in self.MODEL_NAME:
            self.tokenizer.add_special_tokens({
                "bos_token": "</s>", "eos_token": "</s>", "unk_token": "<unk>",
                "pad_token": "<pad>", "mask_token": "<mask>"
            })

        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def forward(self, **kwargs):
        return self.model(**kwargs)

    @staticmethod
    def __clean(x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        )

        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = soynlp.normalizer.repeat_normalize(x, num_repeats=2)
        return x

    def __encode(self, x):
        if "kogpt2" in self.MODEL_NAME:
            encode = self.tokenizer.encode(
                self.__clean(x),
                padding="max_length",
                max_length=self.MAX_LENGTH - 1
            ) + [self.tokenizer.eos_token_id]
        else:
            encode = self.tokenizer.encode(
                self.__clean(x),
                padding="max_length",
                max_length=self.MAX_LENGTH,
            )

        return encode

    def prepare_data(self) -> None:
        # 웹에서 데이터 받아오기
        train_raw_data = pd.read_csv(self.TRAIN_SET_URL, sep="\t")
        test_raw_data = pd.read_csv(self.TEST_SET_URL, sep="\t")

        # 텍스트 인코딩
        train_raw_data["comments"] = train_raw_data["comments"].map(self.__encode)
        test_raw_data["comments"] = test_raw_data["comments"].map(self.__encode)

        self.train_set, self.valid_set = train_test_split(train_raw_data, test_size=0.1)
        self.test_set = test_raw_data

    @staticmethod
    def __dataloader(df, shuffle: bool = False):
        return DataLoader(
            TensorDataset(
                torch.tensor(df["comments"].to_list(), dtype=torch.long),
                torch.tensor(df["hate"].to_list(), dtype=torch.long)
            ),
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self.__dataloader(self.train_set, True)

    def val_dataloader(self):
        return self.__dataloader(self.valid_set)

    def test_dataloader(self):
        return self.__dataloader(self.test_set)

    def __step(self, batch, batch_idx):
        data, labels = batch
        output = self.forward(input_ids=data, labels=labels)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            "loss": loss,
            "y_true": y_true,
            "y_pred": y_pred
        }

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx)

    def __epoch_end(self, outputs, state="train"):
        loss = torch.tensor(0, dtype=torch.float)
        y_true, y_pred = [], []

        for i in outputs:
            loss += i["loss"].cpu().detach()
            y_true += i["y_true"]
            y_pred += i["y_pred"]

        loss = loss / len(outputs)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, labels=np.unique(y_pred), zero_division=1)
        rec = recall_score(y_true, y_pred, labels=np.unique(y_pred), zero_division=1)
        f1 = f1_score(y_true, y_pred, labels=np.unique(y_pred), zero_division=1)

        print(f"[Epoch {self.trainer.current_epoch} {state.upper()}]",
              f"Loss={loss}, Acc={acc}, Prec={prec}, Rec={rec}, F1={f1},",
              "CM={}".format(str(cm).replace("\n", "")))

        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1}

    def training_epoch_end(self, outputs):
        self.__epoch_end(outputs, state="train")

    def validation_epoch_end(self, outputs):
        self.__epoch_end(outputs, state="val")

    def test_epoch_end(self, outputs):
        self.__epoch_end(outputs, state="test")

    def configure_optimizers(self):
        if self.OPTIMIZER == "adam":
            optimizer = Adam(self.parameters(), lr=self.LEARNING_RATE)
        elif self.OPTIMIZER == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.LEARNING_RATE)
        elif self.OPTIMIZER == "sgd":
            optimizer = SGD(self.parameters(), lr=self.LEARNING_RATE)
        else:
            raise NotImplementedError(f"'{self.OPTIMIZER}' is not available.")

        scheduler = ExponentialLR(optimizer, gamma=self.GAMMA)

        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }

    def infer(self, x):
        return torch.softmax(
            self(self.__encode(x)).logits,
            dim=-1
        )
