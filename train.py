import pandas as pd
from model import train_model, get_model_stats

print("Loading dataset...")
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print("Starting training process...")
train_model(df, force=True)

stats = get_model_stats()
if stats:
    print("\n" + "="*40)
    print("FINAL TRAINING RESULTS")
    print("="*40)
    print(f"Test Loss:        {stats.get('loss', 0):.4f}")
    print(f"Base Accuracy:    {stats.get('base_accuracy', 0)*100:.2f}%")
    print(f"Final Accuracy:   {stats['accuracy']*100:.2f}% (with Threshold {stats['threshold']:.2f})")
    print(f"ROC AUC:          {stats['roc_auc']:.4f}")
    print(f"Stroke Precision: {stats['stroke_precision']*100:.2f}%")
    print(f"Stroke Recall:    {stats['stroke_recall']*100:.2f}%")
    print("="*40)

print("Model trained successfully!")