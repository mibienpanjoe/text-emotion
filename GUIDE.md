# Traffic sign recognition

A camera sees one cropped traffic sign. The program answers with the sign's name, plus how sure it is. That answer is what a small robot or a driver-assist demo would use to decide "stop", "yield", or "speed limit 50".

The repository root is the project. This file is the study notes. [README.md](README.md) is the short public overview. Read a section here, then run the matching command.

## What you will have at the end

- A dataset of real German traffic signs (GTSRB), 43 classes.
- A convolutional neural network trained on those photos.
- A test accuracy, a per-class report, and a confusion matrix.
- A command that classifies one new image.

Finding the sign inside a full street photo is a different problem (object detection). We start from photos where the sign is already the subject, which is the right size for two days.

## Map of the files

| File | Role |
|---|---|
| `download_data.py` | Fetch the two Parquet tables into `data/hf/` |
| `class_names.py` | The 43 names, in the official id order (0 = Speed limit 20, 14 = Stop, …) |
| `data.py` | Resize, augment, and batch the images |
| `model.py` | The neural network |
| `explore.py` | Look at the data before training |
| `train.py` | Fit the network and save the best weights |
| `evaluate.py` | Score those weights on images the network never trained on |
| `predict.py` | Classify one photo |

`data/` holds the downloaded images. `outputs/` holds the plot, the saved weights, and the test report. Both are listed in `.gitignore`, so they stay on this machine when the code is published. The plots and the accuracy numbers belong in a write-up. The raw files do not.

## 1. The learning problem

Each example is a pair `(image, class id)`. The class id is an integer from 0 to 42. The network does not output the name directly. It outputs 43 numbers, called **logits**. The largest logit is the predicted class. A **softmax** turns those logits into probabilities that add up to 1, which is what `predict.py` prints.

This is **supervised classification**: every training photo already has the correct label. The dataset authors labeled them. We do not invent labels.

## 2. Libraries, and why each one is here

**PyTorch** (`torch`) is the framework that trains the network.

- A **tensor** is a multi-dimensional array that can live on CPU or GPU and can remember how it was computed.
- **Autograd** records those computations so `loss.backward()` can compute a gradient for every trainable weight.
- `torch.nn` holds layers (a linear layer, a convolution) and losses.
- `torch.optim` holds optimizers such as Adam, which apply those gradients to the weights.

**torchvision** sits on top of PyTorch for images.

- `datasets.GTSRB` downloads the benchmark and returns `(image, label)` pairs.
- `transforms` resizes, augments, and converts a PIL image into a tensor.
- `models.mobilenet_v2` is a published convolutional architecture with weights already trained on ImageNet.

**Pillow** reads image files. You rarely call it yourself; torchvision opens each file with it.

**matplotlib** draws the sample grid and the confusion matrix.

**scikit-learn** is not the trainer. We use only its metrics: precision, recall, and the confusion matrix. Training stays in PyTorch.

## 3. The dataset

GTSRB (German Traffic Sign Recognition Benchmark) is a public dataset from Ruhr University Bochum, published for driver-assistance research. The usual PyTorch path is `torchvision.datasets.GTSRB`, which downloads a zip of `.ppm` photos from `sid.erda.dk`. That server did not accept a connection from this machine, so the project uses the same benchmark as published on Hugging Face (`tanganke/gtsrb`).

The files in `data/hf/` are **Parquet** tables, a columnar file format. **PyArrow** reads them. Each row is an image (raw bytes) and an integer label. Pillow turns those bytes back into a photo inside `SignDataset`.

Counts in this copy:

- 26,640 training photos
- 12,630 test photos
- 43 classes, ids 0 through 42

The classic zip has about 39,000 training photos. This release is smaller and is the one we actually train on. The test set size matches the official test set.

The classes are unbalanced. In this training table, Speed limit 30 and Speed limit 50 have 1,500 photos each. Speed limit 20, Curve left, and Straight or left have 150. A single accuracy number can look strong while a rare class is weak, which is why the evaluation report lists every class.

Images are `.ppm` files of different sizes, often small (around 30 to 100 pixels). The network needs one fixed size, so every photo is resized to 64×64.

The official **test split stays untouched** until `evaluate.py`. During training we hold out 20% of the training images as a **validation set**. Validation tells us when to keep a checkpoint. The test set is the number you report, because the network never used it to update weights or to pick a checkpoint.

## 4. From a photo to a tensor

A color photo is height × width × 3 (red, green, blue), values 0–255. The network wants a tensor of shape `3 × 64 × 64`, values roughly centered around 0.

`eval_transform` does three things, in order:

1. **Resize** to 64×64.
2. **ToTensor** turns the PIL image into a float tensor and scales pixels from 0–255 to 0–1. It also moves channels from the last dimension to the first, which is PyTorch's image layout.
3. **Normalize** subtracts the ImageNet mean and divides by the ImageNet standard deviation, separately per channel. MobileNetV2's pretrained filters expect that scale. Skipping it makes the pretrained trunk much less useful.

`train_transform` adds two random changes before that, so each epoch shows a slightly different version of the same sign:

- **RandomRotation(15)** tilts the sign up to 15 degrees. Real cameras are rarely perfectly level.
- **ColorJitter** changes brightness, contrast, and saturation. Signs are photographed in sun, shade, and dusk.

There is no horizontal flip. A mirrored "turn right" is a different sign. Augmentation has to preserve the label.

A **DataLoader** stacks images into a batch. With `BATCH_SIZE = 64`, each training step sees a tensor of shape `64 × 3 × 64 × 64` and a label vector of length 64. Shuffle is on for training, off for validation and test, so the score does not depend on order.

## 5. The network

A **convolutional neural network** slides small filters over the image. Early filters respond to edges and color blobs. Later filters respond to corners, digits, and the triangular or circular outline of a sign.

We do not train those filters from random weights. **Transfer learning** starts from MobileNetV2, which was trained on ImageNet (about 1.2 million everyday photos, 1000 classes). The early filters are already useful for any photograph. We **freeze** them (`requires_grad = False`), so Adam does not change them.

The original last layer predicts 1000 ImageNet classes. We throw that layer away and put in a new `Linear` layer with 43 outputs, one per sign. Only this new layer, and the small dropout layer in front of it, is trained. That is why a few epochs on a CPU are enough.

MobileNetV2 is the small member of the MobileNet family, built for phones and robots. ResNet18 would also work; it is heavier and slower on this CPU.

Forward pass, in one sentence: the frozen trunk turns a 64×64 photo into a feature vector, and the linear layer turns that vector into 43 logits.

## 6. Training

**Loss.** `CrossEntropyLoss` compares the 43 logits to the true class id. It is low when the true class has the largest logit, and high when a wrong class does. Inside the loss, log-softmax is applied for you. You do not softmax before this loss.

**Optimizer.** Adam updates only the weights with `requires_grad = True`. The learning rate `1e-3` is a standard first choice for a new classifier head.

**Epoch.** One epoch means every training image has been seen once. Each batch produces a loss, `backward()` fills gradients, and `step()` changes the weights.

**What we print.**

- Train loss should fall.
- Validation accuracy should rise, then flatten.
- If train accuracy keeps climbing while validation accuracy falls, the model is memorizing. Five epochs with a frozen trunk usually does not do that.

`train.py` writes `outputs/model.pt` whenever validation accuracy improves. The file contains the weights (`state_dict`) and the validation accuracy at that moment. Later scripts load those weights. They do not keep training.

## 7. Evaluation

`evaluate.py` loads the checkpoint, runs the official test set with no gradient and no augmentation, and compares predicted ids with true ids.

**Accuracy** is the fraction of test images where the predicted id equals the true id. It is the headline number.

**Precision** for one class: of the images we called "Stop", how many really were Stop.

**Recall** for one class: of the real Stop images, how many we called Stop.

**Macro average** in the report is the unweighted mean across the 43 classes. It drops if rare classes fail, even when overall accuracy stays high. That is the number to mention next to accuracy.

The **confusion matrix** is a 43×43 grid. Row is the true class, column is the predicted class. A clean model is bright on the diagonal. Off-diagonal spots are the mistakes worth showing in a presentation (speed limits confused with each other is the usual one, because they share a shape and differ by digits).

## 8. One image

`predict.py` opens a file, applies `eval_transform` (never the random training transform), adds a batch dimension with `unsqueeze(0)`, and runs the network. Softmax turns logits into probabilities. The top three are printed so a near-tie is visible, not only the winner.

## Commands

From this directory:

```bash
python3 download_data.py
python3 explore.py
python3 train.py --epochs 5
python3 evaluate.py
python3 predict.py path/to/a_sign.jpg
```

`download_data.py` writes `data/hf/train.parquet` and `data/hf/test.parquet`. If those files are already present, it leaves them alone.

## How to talk about it

One possible closing sentence for the presentation:

> We trained a MobileNetV2 classifier on GTSRB. The convolutional features come from ImageNet and stay frozen. A new 43-way layer is trained with cross-entropy. On the official test set the accuracy is X, and the main confusions are between similar speed-limit signs.

Fill in X after `evaluate.py`.

## Glossary

| Word | Meaning in this project |
|---|---|
| Class | One sign type, an integer 0–42 |
| Logit | Raw score for one class, before probabilities |
| Softmax | Turns logits into probabilities that sum to 1 |
| Epoch | One full pass over the training set |
| Batch | The 64 images used for one weight update |
| Loss | Cross-entropy; how wrong the logits are |
| Validation set | 20% of train, used only to pick the checkpoint |
| Test set | Official GTSRB test images, used once at the end |
| Transfer learning | Reuse a network trained on a different image task |
| Frozen weights | Filters we do not update |
| Confusion matrix | Counts of true class versus predicted class |
