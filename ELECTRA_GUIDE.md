# ELECTRA Implementation Guide

This guide explains the ELECTRA transformer model implementation added to the mental-health-nlp repository.

## What Was Added

### 1. New Module: `src/electra_risk_detector.py`

A complete ELECTRA-based suicide risk detection implementation featuring:
- Google's ELECTRA transformer model (small or base variants)
- Full integration with HuggingFace Transformers library
- Comprehensive data loading and preprocessing
- Training with HuggingFace Trainer API
- Extensive evaluation metrics and visualizations
- Full safety framework integration (same as LSTM model)

**Key Classes:**
- `ELECTRARiskDetector`: Main class for ELECTRA-based detection
- `SuicideDataset`: PyTorch dataset for HuggingFace tokenization
- `TextPreprocessor`: Text cleaning utilities
- `EthicalWarning`: Safety warnings and logging

### 2. Training Script: `examples/train_electra.py`

Command-line interface for training ELECTRA models with:
- Flexible hyperparameter configuration
- Train/evaluate/interactive modes
- Automatic visualization generation
- Prediction saving and error analysis

### 3. Updated README

Added comprehensive documentation for:
- ELECTRA architecture specifications
- Side-by-side comparison with LSTM
- Complete usage examples
- Command-line training options

## Key Differences: LSTM vs ELECTRA

| Feature | LSTM | ELECTRA |
|---------|------|---------|
| **Architecture** | Bidirectional LSTM | Transformer (ELECTRA) |
| **Parameters** | ~847K | ~14M (small) / ~110M (base) |
| **Input Length** | Max 100 tokens | Max 512 tokens |
| **Training Time** | ~10 min (CPU) | ~30 min (GPU recommended) |
| **Target F1** | ~0.90 | ~0.94 |
| **Best For** | Quick prototyping, limited resources | Production, best performance |

## Quick Start

### Prerequisites

```bash
# Install dependencies (transformers already in requirements.txt)
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
```

### Basic Training

```bash
python examples/train_electra.py \
    --data_path ./Suicide_Detection.csv \
    --num_epochs 3 \
    --batch_size 16 \
    --save_model \
    --plot
```

### Advanced Usage

```bash
# Use larger ELECTRA-base model
python examples/train_electra.py \
    --data_path ./Suicide_Detection.csv \
    --model_name google/electra-base-discriminator \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --save_model \
    --save_predictions \
    --plot \
    --output_dir ./electra_base_results

# Evaluate existing model + interactive predictions
python examples/train_electra.py \
    --data_path ./Suicide_Detection.csv \
    --eval_only \
    --load_model_path ./saved_model \
    --interactive
```

## Code Examples

### Training from Python

```python
from src.electra_risk_detector import ELECTRARiskDetector

# Initialize
detector = ELECTRARiskDetector(
    model_name="google/electra-small-discriminator"
)

# Load data
df, labels = detector.load_data('Suicide_Detection.csv')

# Prepare datasets
train_ds, val_ds, test_ds = detector.prepare_datasets(
    df['cleaned_text'],
    labels
)

# Train
detector.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_epochs=3
)

# Evaluate
results = detector.evaluate_comprehensive(test_ds, df['cleaned_text'])

# Visualize
detector.plot_confusion_matrix(results['labels'], results['predictions'])
detector.plot_roc_and_pr_curves(results['labels'], results['probabilities'])

# Save
detector.save_model('./my_electra_model')
```

### Making Predictions

```python
from src.electra_risk_detector import ELECTRARiskDetector

# Load trained model
detector = ELECTRARiskDetector()
detector.load_model('./my_electra_model')

# Make prediction
text = "I feel hopeless and don't know what to do"
prediction, confidence = detector.predict(text)

# Results
# prediction: 0 (non-suicide) or 1 (suicide risk)
# confidence: probability of predicted class

# Automatic safety protocols trigger for high-risk predictions
```

## Dataset Information

### Required Format

The code expects a CSV file with these columns:
- `text`: The post/message text
- `class`: Either "suicide" or "non-suicide"

### Recommended Dataset

**Kaggle Suicide Watch Dataset:**
- URL: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
- Records: ~232,000 posts
- Source: Reddit r/SuicideWatch and other subreddits
- Balanced: 50/50 split between classes

## Performance Expectations

Based on literature and this implementation:

| Model | F1 Score | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| LSTM | ~0.90 | ~0.88 | ~0.92 | ~0.95 |
| ELECTRA-small | ~0.94 | ~0.93 | ~0.95 | ~0.97 |
| ELECTRA-base | ~0.95 | ~0.94 | ~0.96 | ~0.98 |

*Note: Actual performance depends on training data, hyperparameters, and computational resources.*

## Safety Features

Both LSTM and ELECTRA implementations include:

1. **Ethical Warnings**: Displayed on initialization
2. **High-Risk Logging**: Automatic logging of high-confidence risk predictions
3. **Crisis Resources**: Displayed for high-risk predictions
4. **Human Oversight**: Emphasis on required human review
5. **Audit Trails**: Comprehensive logging for safety monitoring

## Computational Requirements

### ELECTRA-small
- **GPU Memory**: ~4GB
- **Training Time**: ~30 minutes (3 epochs, batch size 16)
- **CPU Fallback**: Possible but slow (~3 hours)

### ELECTRA-base
- **GPU Memory**: ~8GB
- **Training Time**: ~1 hour (3 epochs, batch size 8)
- **CPU Fallback**: Not recommended (~12+ hours)

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python examples/train_electra.py --batch_size 8  # or 4

# Use smaller model
python examples/train_electra.py --model_name google/electra-small-discriminator

# Reduce sequence length
python examples/train_electra.py --max_length 256
```

### Slow Training
```bash
# Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Reduce logging frequency
python examples/train_electra.py --logging_steps 500
```

### Import Errors
```bash
# Reinstall transformers
pip install --upgrade transformers torch

# Verify installation
python -c "from transformers import AutoTokenizer; print('OK')"
```

## Source Code References

The ELECTRA implementation is based on code from:
- Original location: `/Users/blai90/Documents/Repositories/Suicide/nlp_project.py`
- Jupyter notebook: `/Users/blai90/Documents/Repositories/Suicide/NLP_Project.ipynb`

Key improvements made:
1. **Modular design**: Separated into classes for reusability
2. **Safety integration**: Added comprehensive ethical frameworks
3. **Production-ready**: Command-line interface and proper error handling
4. **Documentation**: Extensive docstrings and usage examples
5. **Evaluation**: Comprehensive metrics and visualization tools

## Next Steps

1. **Train baseline model**: Start with ELECTRA-small on a subset
2. **Evaluate performance**: Compare with LSTM baseline
3. **Tune hyperparameters**: Adjust learning rate, epochs, batch size
4. **Scale up**: Use ELECTRA-base for best results
5. **Deploy responsibly**: Always with human oversight

## Additional Resources

- **ELECTRA Paper**: https://arxiv.org/abs/2003.10555
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Suicide Watch Dataset**: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
- **Crisis Resources**: https://suicidepreventionlifeline.org

---

**Important Reminder**: This is for research and educational purposes only. Never deploy for clinical use without appropriate ethical approval, safety protocols, and human oversight.
