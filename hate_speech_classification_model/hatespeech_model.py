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
from transformers import ElectraForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class HateSpeechClassifier(pl.LightningModule):
    def __init__(self, hyper_parameter: dict):
        super().__init__()

        ### 데이터셋 주소 ###
        self.TRAIN_SET_URL = "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/train.tsv"
        self.DEV_SET_URL = "https://raw.githubusercontent.com/2021-hknu-cd-hate-speech-classification/Curse-detection-data/master/dataset.txt"

        ### 하이퍼파라미터 ###
        self.MAX_LENGTH = hyper_parameter["max_length"] or 150
        self.LEARNING_RATE = hyper_parameter["lr"] or 5e-6
        self.EPOCHS = hyper_parameter["epochs"] or 5
        self.MODEL_NAME = hyper_parameter["model"] or "beomi/KcELECTRA-base"
        self.OPTIMIZER = hyper_parameter["optimizer"] or "adamw"
        self.GAMMA = hyper_parameter["gamma"] or 0.5

        ### 사용할 모델 ###
        self.electra = ElectraForSequenceClassification \
            .from_pretrained(self.MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def forward(self, **kwargs):
        return self.electra(**kwargs)

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
        return self.tokenizer.encode(
            self.__clean(x),
            padding="max_length",
            max_length=self.MAX_LENGTH,
            truncation=True
        )

    def prepare_data(self) -> None:
        # 웹에서 데이터 받아오기
        train_raw_data = pd.read_csv(self.TRAIN_SET_URL, sep="\t")
        dev_raw_data = pd.read_csv(self.DEV_SET_URL, sep="\t")

        # 텍스트 인코딩
        train_raw_data["comments"] = train_raw_data["comments"].map(
            self.__encode)
        dev_raw_data["comments"] = dev_raw_data["comments"].map(self.__encode)

        # hate 열의 데이터를 텍스트에서 숫자로 변경
        # none은 0, offensive와 hate가 1임
        train_raw_data["hate"] = train_raw_data["hate"].replace(
            ["none", "offensive", "hate"], [0, 1, 1])

        # 안쓰는 열 제거
        del train_raw_data["contain_gender_bias"]
        del train_raw_data["bias"]

        self.train_set, self.valid_set = train_test_split(
            train_raw_data, test_size=0.1)
        self.test_set = dev_raw_data

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

        for i in outputs:
            loss += i["loss"].cpu().detach()
        loss = loss / len(outputs)

        y_true, y_pred = [], []

        for i in outputs:
            y_true += i["y_true"]
            y_pred += i["y_pred"]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.log(state + "_loss", float(loss), on_epoch=True, prog_bar=True)
        self.log(state + '_acc', acc, on_epoch=True, prog_bar=True)
        self.log(state + '_precision', prec, on_epoch=True, prog_bar=True)
        self.log(state + '_recall', rec, on_epoch=True, prog_bar=True)
        self.log(state + '_f1', f1, on_epoch=True, prog_bar=True)

        print(f"[Epoch {self.trainer.current_epoch} {state.upper()}]",
              f"Loss={loss}, Acc={acc}, Prec={prec}, Rec={rec}, F1={f1}")

        return {"loss": loss}

    def train_epoch_end(self, outputs):
        return self.__epoch_end(outputs, state="train")

    def validation_epoch_end(self, outputs):
        return self.__epoch_end(outputs, state="val")

    def test_epoch_end(self, outputs):
        return self.__epoch_end(outputs, state="test")

    def configure_optimizers(self):
        if self.OPTIMIZER is "adamw":
            optimizer = AdamW(self.parameters(), lr=self.LEARNING_RATE)
        elif self.OPTIMIZER is "sgd":
            optimizer = SGD(self.parameters(), lr=self.LEARNING_RATE)
        else:
            optimizer = Adam(self.parameters(), lr=self.LEARNING_RATE)

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