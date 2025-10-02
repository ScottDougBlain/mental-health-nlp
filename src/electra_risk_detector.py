"""
Mental Health NLP Safety Demo: ELECTRA-based Suicide Risk Detection

An implementation of transformer-based suicide risk detection using Google's ELECTRA model.
This module demonstrates state-of-the-art NLP approaches for mental health text classification
with comprehensive safety frameworks.

NOTE: Model requires training with appropriate data.
Target performance goals: ~0.94 F1 score (based on literature benchmarks).

IMPORTANT: This is for research and educational purposes only. Not for clinical use.
Always direct individuals in crisis to professional mental health resources.
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import re
import unicodedata

# Import safety guidelines from existing module
try:
    from .safety_guidelines import SafetyGuidelines, CRISIS_RESOURCES
except ImportError:
    from safety_guidelines import SafetyGuidelines, CRISIS_RESOURCES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EthicalWarning:
    """Class to handle ethical warnings and safety checks for transformer models."""

    @staticmethod
    def display_usage_warning():
        """Display important usage warnings."""
        warning_text = """
        ⚠️  IMPORTANT ETHICAL AND SAFETY NOTICE ⚠️

        This ELECTRA-based suicide risk detection model is for RESEARCH and EDUCATIONAL purposes only.

        DO NOT USE for:
        - Clinical diagnosis or treatment decisions
        - Automated intervention systems
        - Unsupervised content moderation
        - Any life-critical applications

        ALWAYS:
        - Direct individuals in crisis to professional resources
        - Ensure human oversight for any risk assessments
        - Respect privacy and obtain appropriate consent
        - Follow applicable laws and ethical guidelines

        Crisis Resources:
        - US: National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        - International: https://findahelpline.com

        By using this code, you acknowledge these limitations and responsibilities.
        """
        print(warning_text)


class TextPreprocessor:
    """Text preprocessing utilities for suicide risk detection."""

    @staticmethod
    def unicode_to_ascii(s: str) -> str:
        """Convert unicode string to ascii."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(s: str) -> str:
        """Normalize text for model input."""
        s = TextPreprocessor.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text


class SuicideDataset(Dataset):
    """PyTorch Dataset for suicide risk detection with ELECTRA tokenization."""

    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.

        Args:
            texts: Text data
            labels: Binary labels (0: non-suicide, 1: suicide)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ELECTRARiskDetector:
    """ELECTRA-based suicide risk detector with comprehensive safety framework."""

    def __init__(
        self,
        model_name: str = "google/electra-small-discriminator",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize ELECTRA risk detector.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        # Display ethical warning
        EthicalWarning.display_usage_warning()

        self.model_name = model_name
        self.max_length = max_length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

        # Initialize safety guidelines
        self.safety_guidelines = SafetyGuidelines()

        logger.info(f"Initialized {model_name} for suicide risk detection")

    def load_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load suicide detection dataset from CSV.

        Args:
            csv_path: Path to CSV file with 'text' and 'class' columns

        Returns:
            Tuple of (DataFrame with features, Series with labels)
        """
        logger.info(f"Loading data from {csv_path}")

        df = pd.read_csv(csv_path)

        # Basic validation
        required_cols = ['text', 'class']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Remove any unnecessary columns if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # Convert labels to binary
        df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})

        # Clean text
        df['cleaned_text'] = df['text'].apply(TextPreprocessor.clean_text)

        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Class distribution: {df['class'].value_counts().to_dict()}")

        return df, df['label']

    def prepare_datasets(
        self,
        X: pd.Series,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.5,
        random_state: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation, and test datasets.

        Args:
            X: Text features
            y: Labels
            test_size: Fraction for test set
            val_size: Fraction of remaining for validation
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Split into train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Create datasets
        train_dataset = SuicideDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = SuicideDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = SuicideDataset(X_test, y_test, self.tokenizer, self.max_length)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = './results',
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 100
    ):
        """
        Train the ELECTRA model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save results
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            logging_steps: Logging frequency
        """
        logger.info("Starting model training")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # Train
        self.trainer.train()

        # Evaluate
        eval_results = self.trainer.evaluate()
        logger.info("Validation Results:")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value:.4f}")

    def evaluate_comprehensive(
        self,
        test_dataset: Dataset,
        X_test: pd.Series
    ) -> Dict:
        """
        Comprehensive evaluation with visualizations and safety checks.

        Args:
            test_dataset: Test dataset
            X_test: Test text data (for error analysis)

        Returns:
            Dictionary of evaluation results
        """
        logger.info("Running comprehensive evaluation")

        # Get predictions
        test_predictions = self.trainer.predict(test_dataset)
        test_preds = np.argmax(test_predictions.predictions, axis=1)
        test_labels = test_predictions.label_ids
        test_probs = torch.softmax(
            torch.from_numpy(test_predictions.predictions),
            dim=1
        ).numpy()

        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary'
        )
        roc_auc = roc_auc_score(test_labels, test_probs[:, 1])

        # Print results
        print("\n" + "="*60)
        print("ELECTRA Model - Test Set Results")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            test_labels, test_preds,
            target_names=['Non-Suicide', 'Suicide']
        ))

        # Safety checks - flag high-risk predictions
        high_risk_count = np.sum((test_preds == 1) & (test_probs[:, 1] > 0.8))
        logger.warning(f"High-risk predictions detected: {high_risk_count}")
        logger.warning("Ensure human oversight for all risk assessments")

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': test_preds,
            'probabilities': test_probs,
            'labels': test_labels,
            'texts': X_test.values
        }

        return results

    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Suicide', 'Suicide'],
            yticklabels=['Non-Suicide', 'Suicide']
        )
        plt.title('Confusion Matrix - ELECTRA Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Add percentages
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                plt.text(
                    j + 0.5, i + 0.7,
                    f'({percentage:.1f}%)',
                    ha='center', va='center',
                    fontsize=9, color='gray'
                )

        plt.tight_layout()
        plt.savefig('electra_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Confusion matrix saved to electra_confusion_matrix.png")

    def plot_roc_and_pr_curves(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray
    ):
        """Plot ROC and Precision-Recall curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)

        ax1.plot(
            fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - ELECTRA')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        pr_auc = auc(recall, precision)

        ax2.plot(
            recall, precision, color='darkgreen', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})'
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve - ELECTRA')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('electra_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("ROC and PR curves saved to electra_curves.png")

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Make prediction on new text with safety protocols.

        Args:
            text: Input text

        Returns:
            Tuple of (prediction, confidence)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)

        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prediction = np.argmax(probs)
        confidence = probs[prediction]

        # Safety check and logging
        if prediction == 1 and confidence > 0.8:
            logger.warning(f"HIGH RISK prediction (confidence: {confidence:.3f})")
            logger.warning("CRISIS RESOURCES SHOULD BE DISPLAYED")
            print("\n" + "="*60)
            print("⚠️  CRISIS RESOURCES")
            print("="*60)
            print("US: National Suicide Prevention Lifeline: 988")
            print("Crisis Text Line: Text HOME to 741741")
            print("International: https://findahelpline.com")
            print("="*60 + "\n")

        return int(prediction), float(confidence)

    def save_model(self, path: str):
        """Save trained model and tokenizer."""
        logger.info(f"Saving model to {path}")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved successfully")

    def load_model(self, path: str):
        """Load trained model and tokenizer."""
        logger.info(f"Loading model from {path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Model loaded successfully")


def demo_electra_training():
    """Demo script showing ELECTRA model training."""
    print("\n" + "="*60)
    print("ELECTRA Suicide Risk Detection - Demo")
    print("="*60)
    print("\nNOTE: This is a demonstration. Actual training requires:")
    print("1. Kaggle Suicide Watch dataset")
    print("2. Appropriate computational resources (GPU recommended)")
    print("3. Ethical approval for research use")
    print("\nExample usage:")
    print("""
    # Initialize detector
    detector = ELECTRARiskDetector(
        model_name="google/electra-small-discriminator"
    )

    # Load and prepare data
    df, labels = detector.load_data('Suicide_Detection.csv')
    train_ds, val_ds, test_ds = detector.prepare_datasets(
        df['cleaned_text'], labels
    )

    # Train model
    detector.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=3,
        batch_size=16
    )

    # Evaluate
    results = detector.evaluate_comprehensive(test_ds, df['cleaned_text'])
    detector.plot_confusion_matrix(results['labels'], results['predictions'])
    detector.plot_roc_and_pr_curves(results['labels'], results['probabilities'])

    # Make predictions
    text = "Sample text for prediction"
    prediction, confidence = detector.predict(text)
    print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")

    # Save model
    detector.save_model('./saved_electra_model')
    """)


if __name__ == "__main__":
    demo_electra_training()
