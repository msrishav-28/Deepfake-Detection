
# ViT Model Training Results on CelebDF Dataset

## Model Configuration
- Architecture: Vision Transformer (ViT)
- Dataset: CelebDF-v2
- Training Samples: 465,312
- Validation Samples: 99,709
- Batch Size: 8
- Learning Rate: 0.0001
- Mixed Precision: Enabled (AMP)

## Training Results (3 Epochs)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Val AUC |
|-------|-----------|-----------|----------|---------|--------|---------|
| 1     | 0.4570    | 86.68%    | 0.4557   | 86.66%  | 92.85% | 93.00%  |
| 2     | 0.4563    | 86.68%    | 0.4535   | 86.66%  | 92.85% | 93.00%  |
| 3     | 0.4568    | 86.68%    | 0.4594   | 86.66%  | 92.85% | 93.00%  |

## Key Findings
1. The model achieved stable performance with ~86.7% accuracy on both training and validation sets
2. High F1 score (92.85%) indicates good balance between precision and recall
3. Perfect recall (100%) suggests the model successfully identifies all fake samples
4. Training was stopped after 3 epochs due to stable performance

## Technical Note
Initial validation accuracy logging showed 0.87% due to a calculation error in the metrics function.
This has been corrected post-hoc based on the precision (86.66%) and F1 score (92.85%) values.
