# Traffic sign recognition

A camera sees one cropped traffic sign. This project names it, and reports how sure the model is. That label is what a small robot or a driver-assist demo can turn into an action: stop, yield, or a speed limit.

The model is a MobileNetV2 network. Its convolutional layers stay as they were trained on ImageNet. A new final layer learns the 43 sign classes of the German Traffic Sign Recognition Benchmark (GTSRB).

The study notes, with the libraries and the training concepts, are in [GUIDE.md](GUIDE.md).

## Layout

| File | Role |
|---|---|
| `download_data.py` | Fetch the dataset into `data/hf/` |
| `explore.py` | Class counts and a contact sheet of the signs |
| `data.py` | Load, augment, and batch the photos |
| `model.py` | The network |
| `train.py` | Train and save the best weights |
| `evaluate.py` | Accuracy, per-class report, confusion matrix |
| `predict.py` | Classify one image |
| `class_names.py` | The 43 names, in official id order |

`data/` and `outputs/` are created on your machine and are listed in `.gitignore`. The dataset is a few hundred megabytes. The checkpoint and plots are produced by the scripts.

## Setup

Python 3.10 or newer. PyTorch is required. On a machine that already has it:

```bash
python3 -m pip install -r requirements.txt
python3 download_data.py
```

`download_data.py` skips a file that is already downloaded. This copy has 26,640 training photos and 12,630 test photos, 43 classes.

## Run

```bash
python3 explore.py
python3 train.py --epochs 5
python3 evaluate.py
python3 predict.py path/to/a_sign.jpg
```

Training writes `outputs/model.pt` whenever validation accuracy improves. Evaluation reads that file and writes `outputs/classification_report.txt` and `outputs/confusion_matrix.png`. The test images are not used to choose the checkpoint.

There is no GPU requirement. Five epochs on a CPU is the intended first run.

## After the presentation

Natural next steps, without changing the problem:

- Train the convolutional layers as well, with a smaller learning rate.
- Compare this frozen MobileNetV2 with a small network trained from scratch.
- Show the per-class errors, especially speed limits, which share a shape and differ by the digits.
