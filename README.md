# Brain Tumor MRI Classifier

A PyTorch project for four-class brain MRI classification: glioma, meningioma, no tumour, and pituitary tumour.

The project compares custom convolutional neural networks with frozen-backbone transfer learning, evaluates eight hyperparameter configurations, and provides checkpoint-based training and standalone inference.

## Technical features

- **Architecture comparison:** original CNN, compact CNN, and pretrained ResNet18.
- **Transfer learning:** frozen ResNet18 feature extractor with a trainable linear classification head.
- **Hyperparameter search:** eight configurations, five epochs per configuration, batch size 32.
- **Optimization:** AdamW and momentum-based SGD with weight decay.
- **Regularization:** dropout in the custom CNNs and weight decay.
- **Learning-rate scheduling:** ReduceLROnPlateau monitoring validation loss.
- **Checkpointing:** best-model selection and epoch-boundary resume with model, optimizer, scheduler, and random states.
- **Data auditing:** SHA-256 hashing of decoded RGB images to identify exact duplicates.
- **Evaluation:** accuracy, per-class precision/recall/F1, macro-F1, and confusion matrix.
- **Inference:** command-line prediction using exported model weights and preprocessing metadata.

## Results

| Metric | Recorded result |
|---|---:|
| Selected experiment | `original_adamw_lr0003` |
| Architecture | Original CNN |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Filtered validation accuracy | 88.63% |
| Filtered test images | 1,468 |
| Filtered test accuracy | 84.81% |
| Filtered test macro-F1 | 0.836 |

These results use retrospectively filtered evaluation subsets. See the evaluation limitations below.

![Confusion matrix](results/confusion_matrix.png)

![Model comparison](results/model_comparison.png)

![Training curves](results/training_curves.png)

## Dataset and attribution

**Dataset:** Brain Tumor MRI Dataset  
**Creator/curator:** Masoud Nickparvar  
**Source:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  
**Licence listed on the current dataset page:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The dataset card credits the figshare, SARTAJ, and Br35H datasets as underlying sources.

The local dataset used in this project contained:

- Training folder: 5,600 images, 1,400 per class.
- Testing folder: 1,600 images, 400 per class.
- A stratified 20% split of the training folder was reserved for validation using seed 42.

Project preprocessing includes resizing and normalization. Exact-image auditing and retrospective evaluation filtering were applied as described below.

Dataset images are not redistributed in this repository. Download them from the source page. The dataset licence does not automatically license this repository's code.

## Run a prediction

From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run inference on an image:

```bash
python predict_brain_tumor.py "path/to/image.jpg"
```

The exported model is stored at:

```text
exported_model/brain_tumor_model.pth
```

The script returns the predicted class and softmax scores. These scores are not calibrated clinical probabilities.

Model reload and separate-process prediction consistency checks passed during development. Installation in a fresh environment has not yet been verified.

## Notebook

The organized workflow is available in:

```text
notebooks/experiment_log.ipynb
```

The notebook opens in **review mode** by default:

- `RUN_PIPELINE = False`
- `RUN_TRAINING = False`

Running its cells in order in review mode displays saved reports and figures without starting training.

Optional evaluation replay requires the original dataset and all experiment checkpoints. Configure `WORK_DIR` and follow the notebook instructions before enabling the pipeline.

Full training checkpoints are excluded from this repository. The exported inference model alone cannot resume all eight experiments.

Replay outputs are written separately from the committed results. This notebook organizes the historical experiment; cleaning it did not rerun training or remove the original experiment's methodological limitations.

## Evaluation limitations

Exact duplicates were discovered **after training**.

The audit found the following shared exact-image groups:

- Training/validation: 47.
- Training/testing: 85.
- Validation/testing: 26.

These pairwise counts are not additive unique-image counts.

Validation evaluation excluded exact images present in training. Test evaluation excluded exact images present in either original development split. Repeated images within each evaluation subset were reduced to one copy.

However:

- Original validation data had already influenced checkpoint selection and learning-rate scheduling.
- Training curves show the original validation history.
- Patient overlap and near-duplicates have not been ruled out.
- Five epochs per configuration constitute a limited search.
- A configured scheduler does not imply that learning-rate reductions occurred in every run.

Results are exploratory and should not be presented as a fully leakage-free or clinically validated benchmark.

A future experiment should deduplicate before training, use patient-grouped splits where identifiers are available, and evaluate on an untouched external dataset.

## Tutorial credit

Initial tutorial:

https://www.youtube.com/watch?v=mGJ6OZmR-Xk

The project was extended with architecture comparisons, hyperparameter experiments, stateful checkpointing, duplicate auditing, evaluation reports, model export, and standalone inference.

## Intended use

Educational research only. Not for clinical diagnosis.