# Text emotion classification

A short English message goes in. The program names the emotion it expresses, and reports how sure it is. The six classes are sadness, joy, love, anger, fear, and surprise.

The model is a word embedding followed by an LSTM. The embedding and the recurrent layer are trained on the public Emotion dataset (16,000 training messages, 2,000 validation, 2,000 test). The test messages are not used to choose the checkpoint.

The study notes, with the libraries and the network concepts, are in [GUIDE.md](GUIDE.md).

## Layout

| File | Role |
|---|---|
| `download_data.py` | Fetch the dataset into `data/hf/` |
| `explore.py` | Class counts, example messages, and a bar chart |
| `data.py` | Tokenize, build the vocabulary, and batch the texts |
| `emotions.py` | The six names, in dataset id order |
| `model.py` | Embedding, LSTM, and the emotion scores |
| `train.py` | Train and save the best weights |
| `evaluate.py` | Accuracy, per-class report, confusion matrix |
| `predict.py` | Classify one message |

`data/` and `outputs/` are created on your machine and are listed in `.gitignore`.

## Setup

Python 3.10 or newer, with PyTorch.

```bash
python3 -m pip install -r requirements.txt
python3 download_data.py
```

`download_data.py` skips a file that is already downloaded.

## Run

```bash
python3 explore.py
python3 train.py --epochs 8
python3 evaluate.py
python3 predict.py "I can't believe I got the job"
```

Training writes `outputs/model.pt` whenever validation accuracy improves. That file holds the weights and the vocabulary. Evaluation reads it and writes `outputs/classification_report.txt` and `outputs/confusion_matrix.png`.

There is no GPU requirement. Eight epochs on this CPU took about two minutes.

## Result from the reference run

With seed 42 and the default eight epochs, the checkpoint chosen on validation reaches:

- test accuracy **86.9%**
- macro F1 **0.825**

Surprise is the smallest class (572 training messages) and the weakest on the test report. Joy and love share wording, so a joyful message often keeps a large probability on love as well.

## After the presentation

Natural next steps, without changing the problem:

- Replace the learned embedding with a pretrained one.
- Swap the LSTM for a small pretrained transformer and compare the confusion matrix.
- Look at the messages the model mixes up, especially love versus joy, and anger versus sadness.
