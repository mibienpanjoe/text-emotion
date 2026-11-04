# Text emotion classification

A person writes a short message. The program answers with one of six emotions: sadness, joy, love, anger, fear, or surprise, plus how sure it is.

The repository root is the project. This file is the study notes. [README.md](README.md) is the short public overview. Read a section here, then run the matching command.

## What you will have at the end

- A public dataset of English messages, each already labeled with one emotion.
- A neural network that reads the words in order and predicts that emotion.
- A test accuracy, a per-class report, and a confusion matrix.
- A command that classifies one new sentence.

This is text classification. The model does not write a reply. It chooses one label.

## Map of the files

| File | Role |
|---|---|
| `download_data.py` | Fetch the three Parquet tables into `data/hf/` |
| `emotions.py` | The six names, in official id order (0 = sadness, 1 = joy, …) |
| `data.py` | Tokenize, build the vocabulary, pad, and batch |
| `model.py` | The neural network |
| `explore.py` | Look at the messages before training |
| `train.py` | Fit the network and save the best weights |
| `evaluate.py` | Score those weights on messages the network never trained on |
| `predict.py` | Classify one message |

`data/` holds the downloaded tables. `outputs/` holds the chart, the saved weights, and the test report. Both are listed in `.gitignore`, so they stay on this machine when the code is published.

## 1. The learning problem

Each example is a pair `(text, emotion id)`. The id is an integer from 0 to 5. The network outputs 6 numbers, called **logits**. The largest logit is the predicted emotion. A **softmax** turns those logits into probabilities that add up to 1, which is what `predict.py` prints.

This is **supervised classification**. The dataset already contains the label for every message.

## 2. Libraries, and why each one is here

**PyTorch** (`torch`) trains the network.

- A **tensor** is a multi-dimensional array that can remember how it was computed.
- **Autograd** records those computations so `loss.backward()` can compute a gradient for every trainable weight.
- `torch.nn` holds the embedding, the LSTM, the linear layer, and the loss.
- `torch.optim` holds Adam, which applies those gradients to the weights.

**PyArrow** reads the dataset. Each row in the Parquet table is a message and an integer label.

**matplotlib** draws the class-count bar chart and the confusion matrix.

**scikit-learn** is not the trainer. It only computes precision, recall, and the confusion matrix after prediction.

## 3. The dataset

The Emotion dataset (published with the paper "CARER: Contextualized Affect Representations for Emotion Recognition", and distributed as `dair-ai/emotion`) is a set of English messages collected for emotion classification.

`download_data.py` fetches the standard split:

- 16,000 training messages
- 2,000 validation messages
- 2,000 test messages
- 6 classes, ids 0 through 5

The classes are unbalanced. Joy and sadness are common. Surprise and love are smaller. A single accuracy number can look strong while a rare emotion is weak, which is why the evaluation report lists every class.

The official **test split stays untouched** until `evaluate.py`. The published **validation split** is what `train.py` uses to decide which checkpoint to keep. We do not cut a second validation set out of the training messages.

## 4. From a message to a tensor

A sentence is not a tensor. Four steps make it one.

1. **Tokenize.** `tokenize` lowercases the text and keeps words, numbers, and apostrophes. `"I can't believe it"` becomes `["i", "can't", "believe", "it"]`. Punctuation is dropped. The label does not depend on the comma.

2. **Vocabulary.** Training builds a dictionary from the training messages only. A word seen at least twice gets an integer id. Index 0 is `<pad>`. Index 1 is `<unk>`. A word that is rare, or that appears only in a test message, becomes `<unk>`. Building this dictionary on the test set would leak information.

3. **Encode.** A message becomes a list of ids, cut at 40 tokens. These messages are short, so 40 is enough. An empty result becomes a single `<unk>`, so the LSTM always receives at least one step.

4. **Pad and batch.** Messages in one batch have different lengths. `pad_sequence` extends the shorter ones with `<pad>` so they form a rectangle of shape `batch × longest`. A separate length vector records the real length. The LSTM uses that length to ignore the padding.

A **DataLoader** yields `(token ids, lengths, labels)`. Shuffle is on for training and off for validation and test.

## 5. The network

The class in `model.py` is `EmotionLSTM`. It is three modules, and only those three. There is one LSTM layer, it reads left to right, and nothing is pretrained. Every weight is learned from these messages, which keeps each piece small enough to explain.

```text
word ids  →  Embedding  →  LSTM, one step per word  →  Linear  →  6 logits
(batch, T)   (batch, T, 128)   last hidden (batch, 128)         (batch, 6)
```

`T` is the length of the message after padding. In the reference run the vocabulary has 7,401 rows: one for `<pad>`, one for `<unk>`, and one for every training word seen at least twice.

### Embedding

An embedding is a lookup table of shape `7,401 × 128`. The integer id of a word selects one row. That row is a vector of 128 numbers. At the start the rows are random. Training moves words that play a similar role in these messages toward each other, so "happy" and "glad" can end up nearby if the data supports it. The row for `<pad>` stays the zero vector because it is marked as `padding_idx`. A padding position must not look like a real word.

### Why the order of the words matters

A bag of words would throw the words into one pile and average them. Then "I am happy" and "I am not happy" look similar, because they contain almost the same words. Emotion often depends on order and on a word that happened a few positions earlier ("not", "but", "still"). A recurrent layer exists for that. It reads the vectors one by one and keeps a memory.

### What the LSTM keeps

A plain recurrent layer has one memory vector. At every new word it mixes that memory with the new word and overwrites it. Over a long sentence the early words get multiplied away. That is the vanishing-gradient problem: the signal from "not" at the start can disappear before the end.

An **LSTM** (long short-term memory) keeps two vectors of size 128 at every step:

- The **cell state** is the memory tape. It is added to more than it is multiplied, so a value written early can still be there at the last word.
- The **hidden state** is the part of that memory the layer is willing to show. The next word sees the hidden state, and the final hidden state is what our linear layer reads.

Both start at zero before the first word. After each word they are replaced by a new pair.

### The three gates

At each word the LSTM builds three valves, called gates. Each valve is a small layer that looks at the current word vector and the previous hidden state, then squashes the result with a **sigmoid**. A sigmoid output is between 0 and 1. Multiplied by another vector, 0 means "let nothing through" and 1 means "let all of it through".

- **Forget gate.** How much of the old cell state to keep. Near 0, the previous memory is dropped. Near 1, it is carried forward.
- **Input gate.** How much of a new candidate to write into the cell. The candidate itself is a **tanh** of the current word and the previous hidden state, so its values lie between −1 and 1.
- **Output gate.** How much of the updated cell to reveal as the new hidden state. The cell can remember something that the hidden state does not yet show.

The update, in words: new cell = forget gate × old cell + input gate × candidate. New hidden = output gate × tanh(new cell). PyTorch does this arithmetic inside `nn.LSTM`. We do not code the gates ourselves. The weights of those valves are what Adam learns.

### One sentence, step by step

Take "i feel romantic too", which is a love example in the training set.

1. Tokenize to `["i", "feel", "romantic", "too"]`, then replace each word by its id. Shape: `(4,)`. In a batch of 64 this becomes `(64, T)`.
2. The embedding turns each id into 128 numbers. Shape: `(64, T, 128)`.
3. The LSTM reads "i", updates cell and hidden, then "feel", then "romantic", then "too". Each step uses the memory from the step before.
4. `pack_padded_sequence` tells the LSTM the real length of each message in the batch. Shorter messages are padded with `<pad>` so the batch is a rectangle. Without packing, those pad positions would keep updating the memory after the sentence is over.
5. The LSTM returns a hidden tensor of shape `(num_layers, batch, 128)`. We have one layer, so `hidden[-1]` is `(batch, 128)`: the hidden state after the last real word. The cell state is discarded here. It has already done its job by shaping that hidden state.
6. The linear layer is a matrix of shape `128 × 6` plus a bias of 6. It maps the summary vector to one logit per emotion.

The largest logit wins. Softmax is only for the printed probabilities, not for the loss.

### What this model is not

It is not bidirectional: it never looks ahead at "too" while it is reading "feel". The later words still influence the final hidden state, because that state is produced after them. It is not stacked: there is no second LSTM reading the first LSTM's outputs. It is not a transformer: there is no attention over all words at once. Those are possible upgrades. This version is the one we can draw and defend.

## 6. Training

**Loss.** `CrossEntropyLoss` compares the 6 logits to the true emotion id. It is low when the true class has the largest logit. Log-softmax is inside the loss. You do not softmax before it.

The classes are unbalanced, so the loss uses a **weight per class**. A rare emotion contributes more when it is missed. The weight for a class is `total / (6 * count of that class)`, computed on the training split only.

**Optimizer.** Adam updates every weight: embedding, LSTM, and classifier. The learning rate `1e-3` is a standard first choice.

**Epoch.** One epoch means every training message has been seen once.

**What we print.**

- Train loss should fall.
- Validation accuracy should rise, then flatten.
- If train accuracy keeps climbing while validation accuracy falls, the model is memorizing.

`train.py` writes `outputs/model.pt` whenever validation accuracy improves. The file contains the weights, the vocabulary, and the validation accuracy at that moment. Prediction must use that same vocabulary. A word list built later, or from different texts, would point the ids at the wrong rows of the embedding.

## 7. Evaluation

`evaluate.py` loads the checkpoint, runs the test split with no gradient, and compares predicted ids with true ids.

**Accuracy** is the fraction of test messages where the predicted id equals the true id.

**Precision** for one emotion: of the messages we called "anger", how many really were anger.

**Recall** for one emotion: of the real anger messages, how many we called anger.

**Macro average** is the unweighted mean across the six emotions. It drops if surprise or love fails, even when overall accuracy stays high.

The **confusion matrix** is a 6×6 grid. Row is the true emotion, column is the predicted one. A clean model is bright on the diagonal. The usual confusions are joy with love, and sadness with anger, because the words overlap.

## 8. One message

`predict.py` tokenizes with the saved vocabulary, runs the network, and prints the top three probabilities. There is no padding step, because a single message does not need to be lined up with another one.

## Commands

From this directory:

```bash
python3 download_data.py
python3 explore.py
python3 train.py --epochs 8
python3 evaluate.py
python3 predict.py "I can't believe I got the job"
```

`download_data.py` writes `data/hf/train.parquet`, `validation.parquet`, and `test.parquet`. If those files are already present, it leaves them alone.

## How to talk about it

One possible closing sentence for the presentation:

> We trained an embedding and an LSTM on the Emotion dataset. Each message is a sequence of word ids. The LSTM summarizes that sequence, and a linear layer scores six emotions. On the official test set the accuracy is 86.9%, and the main confusions are between emotions that share wording, such as joy and love.

That 86.9% is the reference run with seed 42 and eight epochs. Macro F1 on the same run is 0.825. Surprise is the rarest training class and the weakest test class.

## Glossary

| Word | Meaning in this project |
|---|---|
| Token | A word after lowercasing |
| Vocabulary | The list of words that receive their own id |
| Unknown token | A word that is not in the vocabulary |
| Padding | Filler ids that make a batch rectangular |
| Embedding | A learned vector of 128 numbers for each word id |
| Recurrent layer | A layer that reads a sequence one step at a time and keeps a memory |
| LSTM | A recurrent layer with a cell state and three gates, so early words can survive |
| Cell state | The LSTM's memory tape, updated by addition at each word |
| Hidden state | The part of that memory exposed to the next word and to the classifier |
| Gate | A sigmoid valve between 0 and 1 that lets information through or blocks it |
| Forget gate | How much of the previous cell state to keep |
| Input gate | How much of the new candidate to write into the cell |
| Output gate | How much of the cell to reveal as the hidden state |
| Vanishing gradient | Early words losing their effect because a plain recurrent memory is repeatedly shrunk |
| Logit | Raw score for one emotion, before probabilities |
| Softmax | Turns logits into probabilities that sum to 1 |
| Epoch | One full pass over the training set |
| Batch | The 64 messages used for one weight update |
| Loss | Weighted cross-entropy; how wrong the logits are |
| Validation set | The published 2,000 messages used to pick the checkpoint |
| Test set | The published 2,000 messages used once at the end |
| Confusion matrix | Counts of true emotion versus predicted emotion |
