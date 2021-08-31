# 모델

## 사용 방법

pip을 이용해 설치

```sh
pip install "git+https://github.com/2021-hknu-cd-hate-speech-classification/model.git"
```

```py
from hate_speech_classification_model import HateSpeechClassifier
```

## 하이퍼파라미터

`__init__` 함수의 `hyper_parameter` 매개 변수에 `dict`형태로 넣어주면 사용할 수 있다.

| 이름           | 설명                           | 값으로 들어갈 수 있는 것                                                    | 기본값               |
| ------------ | ---------------------------- | ----------------------------------------------------------------- | ----------------- |
| `model`      | fine-tuning에 사용될 모델 이름       | `'beomi/KcELECTRA'`, `'monologg/koelectra-base-v3-discriminator'` | `beomi/KcELECTRA` |
| `max_length` | Tokenizer가 사용할 문자열 길이        | `int`형 값.                                                         | `150`             |
| `lr`         | 학습률                          | `float`형 값.                                                       | `5e-6`            |
| `epochs`     | epoch 수                      | `int`형 값.                                                         | `5`               |
| `optimizer`  | 사용할 optimizer의 종류            | `'adam'`, `'adamw'`, `'sgd'`                                      | `adamw`           |
| `gamma`      | Scheduler가 사용할 gamma값        | `float`형 값.                                                       | `0.5`             |
| `batch_size` | DataLoader가 사용할 batch_size 값 | `int`형 값                                                          | `32`              |
