"""
Example script for training ELECTRA-based suicide risk detector.

This script demonstrates how to:
1. Load the Suicide Watch dataset from Kaggle
2. Train an ELECTRA model for risk detection
3. Evaluate performance with comprehensive metrics
4. Save the trained model

Requirements:
- Kaggle dataset: nikhileswarkomati/suicide-watch
- GPU recommended (but not required)
- Ethical approval for research use

NOTE: This is for research and educational purposes only.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.electra_risk_detector import ELECTRARiskDetector
import pandas as pd


def main(args):
    """
    Main training script for ELECTRA model.

    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("ELECTRA Suicide Risk Detection - Training Script")
    print("="*80)

    # Initialize detector
    print(f"\nInitializing ELECTRA model: {args.model_name}")
    detector = ELECTRARiskDetector(
        model_name=args.model_name,
        max_length=args.max_length
    )

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df, labels = detector.load_data(args.data_path)

    # Display data statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Class distribution:")
    print(df['class'].value_counts())
    print(f"\nAverage text length: {df['text'].str.len().mean():.1f} characters")
    print(f"Average word count: {df['text'].str.split().str.len().mean():.1f} words")

    # Prepare datasets
    print("\nPreparing train/validation/test splits...")
    train_dataset, val_dataset, test_dataset = detector.prepare_datasets(
        df['cleaned_text'],
        labels,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Get indices for test set (for later evaluation)
    test_indices = test_dataset.texts.index
    X_test = df.loc[test_indices, 'cleaned_text']

    # Train model
    if not args.eval_only:
        print(f"\nTraining model for {args.num_epochs} epochs...")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")

        detector.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps
        )

        # Save model
        if args.save_model:
            save_path = Path(args.output_dir) / "final_model"
            detector.save_model(str(save_path))
    else:
        # Load existing model
        print(f"\nLoading model from: {args.load_model_path}")
        detector.load_model(args.load_model_path)

    # Comprehensive evaluation
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)

    results = detector.evaluate_comprehensive(test_dataset, X_test)

    # Generate visualizations
    if args.plot:
        print("\nGenerating visualizations...")
        detector.plot_confusion_matrix(results['labels'], results['predictions'])
        detector.plot_roc_and_pr_curves(results['labels'], results['probabilities'])

    # Save predictions
    if args.save_predictions:
        results_df = pd.DataFrame({
            'text': results['texts'],
            'true_label': results['labels'],
            'predicted_label': results['predictions'],
            'confidence': results['probabilities'].max(axis=1),
            'prob_non_suicide': results['probabilities'][:, 0],
            'prob_suicide': results['probabilities'][:, 1],
            'correct': results['labels'] == results['predictions']
        })

        pred_path = Path(args.output_dir) / "predictions.csv"
        results_df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to: {pred_path}")

        # Display interesting cases
        print("\n" + "="*60)
        print("High Confidence Correct Predictions:")
        print("="*60)
        correct_confident = results_df[results_df['correct']].nlargest(
            5, 'confidence'
        )[['text', 'true_label', 'confidence']]
        for idx, row in correct_confident.iterrows():
            print(f"\nText: {row['text'][:100]}...")
            print(f"Label: {row['true_label']}, Confidence: {row['confidence']:.3f}")

        print("\n" + "="*60)
        print("Low Confidence Predictions (Uncertain Cases):")
        print("="*60)
        uncertain = results_df.nsmallest(
            5, 'confidence'
        )[['text', 'true_label', 'predicted_label', 'confidence']]
        for idx, row in uncertain.iterrows():
            print(f"\nText: {row['text'][:100]}...")
            print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}, "
                  f"Confidence: {row['confidence']:.3f}")

    # Test interactive prediction
    if args.interactive:
        print("\n" + "="*80)
        print("INTERACTIVE PREDICTION MODE")
        print("="*80)
        print("Enter text to predict (or 'quit' to exit):")

        while True:
            text = input("\n> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break

            prediction, confidence = detector.predict(text)
            label = "SUICIDE RISK" if prediction == 1 else "No risk detected"
            print(f"\nPrediction: {label}")
            print(f"Confidence: {confidence:.3f}")

    print("\n" + "="*80)
    print("Training and evaluation complete!")
    print("="*80)

    # Summary
    print("\nFinal Results Summary:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")

    if args.save_model:
        print(f"\nTrained model saved to: {Path(args.output_dir) / 'final_model'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ELECTRA model for suicide risk detection"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Suicide_Detection.csv dataset"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/electra-small-discriminator",
        help="HuggingFace model name (default: google/electra-small-discriminator)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging frequency in steps (default: 100)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./electra_results",
        help="Output directory for results (default: ./electra_results)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save trained model"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to CSV"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save plots"
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        help="Path to load model from (for eval_only mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive prediction mode after evaluation"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.eval_only and not args.load_model_path:
        parser.error("--eval_only requires --load_model_path")

    main(args)
