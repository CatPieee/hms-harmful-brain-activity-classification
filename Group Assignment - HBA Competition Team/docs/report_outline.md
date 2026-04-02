# Suggested 2000+ Word Report Structure

## 1. Introduction
- Background of harmful brain activity classification
- Clinical significance
- Why deep learning is suitable
- Brief overview of Kaggle HMS dataset
- Existing mainstream methods: CNN, spectrogram classification, EEG temporal modeling, multimodal fusion

## 2. Dataset and Preprocessing
- Dataset source and task description
- Label format: vote distributions across 6 classes
- EEG preprocessing and caching
- Spectrogram preprocessing and PNG generation
- Why a held-out validation fold is used instead of public test accuracy

## 3. Model Design
- Spectrogram encoder
- EEG encoder
- Gated multimodal fusion
- Output layer and 6-class prediction
- Why multimodal fusion is stronger than a single branch

## 4. Loss Function, Optimizer, and Hyperparameters
- KL divergence for soft labels
- Cross-entropy comparison
- AdamW optimizer
- Learning rate, batch size, epochs, weight decay
- Patient-level GroupKFold split rationale

## 5. Demonstration and Performance
- Baseline training curve
- Different loss functions
- Learning-rate sweep
- Batch-size sweep
- First 100 validation predictions
- Interpret overfitting / generalization behavior

## 6. Critical Analysis
- Why KL divergence works well for vote-distribution labels
- Why validation accuracy may plateau while training accuracy rises
- Why patient-level split is more trustworthy
- Limitations: compute cost, data imbalance, no public labels for Kaggle test set

## 7. Conclusion and Future Work
- Summary of system effectiveness
- Potential improvements: stronger backbones, attention fusion, augmentations, cross-validation averaging, test-time augmentation
