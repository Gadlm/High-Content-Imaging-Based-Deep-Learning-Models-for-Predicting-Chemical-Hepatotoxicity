# High-Content Imaging-Based Deep Learning Models for Predicting Chemical Hepatotoxicity

## Code

- `src/hci_hepatotoxicity/features.py`: extracts cell, nuclear, blue-channel, red-channel, and green-channel image features from Cellpose masks and fluorescence images.
- `src/hci_hepatotoxicity/rf.py`: builds the random forest model.
- `src/hci_hepatotoxicity/knn.py`: builds the k-nearest-neighbor model.
- `src/hci_hepatotoxicity/svm.py`: builds the support-vector-machine model.
- `src/hci_hepatotoxicity/cnn.py`: builds the convolutional neural network model.
- `src/hci_hepatotoxicity/cnn_bilstm_attention.py`: builds the attention-based CNN–BiLSTM model.
- `src/hci_hepatotoxicity/workflow.py`: creates chemical-level profiles, splits chemicals, applies training-only scaling and model-dependent resampling, and aggregates probabilities by chemical.
- `src/hci_hepatotoxicity/applicability.py`: fits and applies the phenotypic support-density and local label-inconsistency applicability domain.
