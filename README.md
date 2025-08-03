# Transformer-Scratch

**Transformer Model built from scratch using PyTorch**

This repository implements a scalable, vanilla Transformer architecture for sequence-to-sequence machine translation from English to Italian. We use the Opus Books dataset from Hugging Face as an example, but you can easily adapt the code to other language pairs or datasets.

---

## Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Model Architecture](#model-architecture)
* [Configuration](#configuration)
* [Training](#training)
* [Evaluation & Metrics](#evaluation--metrics)
* [Inference (Translation)](#inference-translation)
* [Hyperparameters](#hyperparameters)
* [Logging & Checkpoints](#logging--checkpoints)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* 🔄 **Vanilla Transformer from "Attention Is All You Need"**: Multi-head self-attention and position-wise feed-forward networks.
* 📚 **Hugging Face Opus Books dataset**: Preprocessed English–Italian parallel corpus.
* 🛠️ **Modular code base**: Separate modules for data loading (`dataset.py`), model definition (`model.py`), and training loop (`train.py`).
* 💾 **Checkpointing**: Save & resume training, track best validation BLEU.
* ⚙️ **Configurable**: All hyperparameters and paths in `config.py`.

---

## Repository Structure

```text
Transformer-Scratch/
├── config.py               # Hyperparameters, file paths, and training settings
├── dataset.py              # Dataset loader & preprocessing
├── model.py                # Transformer encoder-decoder implementation
├── train.py                # Training and validation loop
├── tokenizer_en.json       # English tokenizer vocabulary
├── tokenizer_it.json       # Italian tokenizer vocabulary
├── README.md               # (you are here)
└── checkpoints/            # Directory for model checkpoints (auto-created)
```

---

## Requirements

* Python 3.8+
* PyTorch 1.10+
* Hugging Face `tokenizers`
* `numpy`, `tqdm`

You can install all dependencies via:

```bash
pip install torch tokenizers numpy tqdm
```

---

## Installation

1. **Clone the repo**

   ```bash
   ```

git clone [https://github.com/1284574/Transformer-Scratch.git](https://github.com/1284574/Transformer-Scratch.git)
cd Transformer-Scratch

````
2. **Create and activate a virtual environment**  
   ```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .\.venv\Scripts\activate  # Windows
````

3. **Install dependencies**

   ```bash
   ```

pip install -r requirements.txt  # if you have created a requirements.txt

# or

pip install torch tokenizers numpy tqdm

````

---

## Data Preparation

We use the Opus Books parallel corpus for English–Italian translation. By default, `dataset.py` will:

1. Download the dataset from Hugging Face (if not already present).
2. Tokenize both English and Italian sentences with the provided JSON vocabularies.
3. Create PyTorch `Dataset` and `DataLoader` objects for training and validation.

If you want to use a different dataset, modify `DATA_PATH`, `SOURCE_LANG`, and `TARGET_LANG` in `config.py` accordingly.

---

## Model Architecture

The core components live in `model.py`:

- **`TransformerEncoder`**: Stacks of multi-head self-attention + feed-forward layers.
- **`TransformerDecoder`**: Masked self-attention + encoder–decoder attention + feed-forward.
- **Positional Encoding**: Adds sinusoidal positional embeddings to token embeddings.
- **`Seq2SeqTransformer`**: Wraps encoder & decoder, handling masks and forward pass.

You can dive into the code comments for detailed line-by-line explanations.

---

## Configuration

All training settings are in `config.py`. Key parameters:

| Parameter          | Description                                   | Default    |
| ------------------ | --------------------------------------------- | ---------- |
| `SRC_VOCAB_PATH`   | Path to English tokenizer JSON                | `tokenizer_en.json` |
| `TGT_VOCAB_PATH`   | Path to Italian tokenizer JSON                | `tokenizer_it.json` |
| `DATA_PATH`        | Directory or HugginFace dataset identifier    | `opus_books`       |
| `BATCH_SIZE`       | Training batch size                           | `64`       |
| `EMB_SIZE`         | Token embedding dimension                     | `512`      |
| `NHEAD`            | Number of attention heads                     | `8`        |
| `FFN_HID_DIM`      | Hidden dimension of feed-forward network      | `2048`     |
| `NUM_ENCODER_LAYERS` | Number of encoder layers                    | `6`        |
| `NUM_DECODER_LAYERS` | Number of decoder layers                    | `6`        |
| `DROPOUT`          | Dropout probability                           | `0.1`      |
| `NUM_EPOCHS`       | Total training epochs                         | `20`       |
| `LEARNING_RATE`    | Initial learning rate                         | `1e-4`     |
| `CHECKPOINT_DIR`   | Directory to save model checkpoints           | `checkpoints/`     |
| `DEVICE`           | `cuda` or `cpu`                               | `'cuda' if torch.cuda.is_available() else 'cpu'` |

Modify these values to experiment quickly without touching the code.

---

## Training

To start training, simply run:

```bash
python train.py \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-4 \
  --checkpoint_dir checkpoints/
````

* Training and validation losses (and BLEU scores) will print on-screen.
* Model checkpoints (one per epoch + best BLEU) will be saved under `checkpoints/`.

You can also resume from a specific checkpoint:

```bash
python train.py --resume checkpoints/epoch_10.pth
```

---

## Evaluation & Metrics

After training, you can compute BLEU on the validation set:

```bash
python train.py \
  --evaluate checkpoints/best_model.pth \
  --batch_size 128
```

This will report corpus-level BLEU using `sacrebleu` under the hood.

---

## Inference (Translation)

To translate a single English sentence into Italian:

```bash
python model.py \
  --mode translate \
  --checkpoint checkpoints/best_model.pth \
  --src "Hello, how are you?" \
  --max_len 50
```

*Output*:

```
> Input: Hello, how are you?
> Translation: Ciao, come stai?
```

You can batch-translate a text file of sentences by passing `--input_file input.txt --output_file output.txt`.

---

## Hyperparameters

Feel free to tweak hyperparameters in `config.py` or via the CLI flags:

* Learning rate schedulers
* Label smoothing
* Weight decay
* Different optimizer (AdamW, Adafactor)

Track experiments with TensorBoard or Weights & Biases by hooking into `train.py`.

---

## Logging & Checkpoints

* **TensorBoard**: call `tensorboard --logdir runs/` after adding `SummaryWriter` hooks in `train.py`.
* **Checkpoints**: model weights + optimizer state saved automatically.

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-awesome-feature`
3. Commit your changes
4. Open a Pull Request

Please adhere to the existing code style and add tests or examples for new features.

---

## License

This project is released under the [GPL-3.0 License](LICENSE).

---

Happy translating! \:it:🚀
