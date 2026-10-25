# SFFR-Net

This repository provides the implementation of Spatial-Frequency Feature Refinement Network (SFFR-Net) for generalizable deepfake detection. The implementation is developed based on DeepfakeBench.

## Method

SFFR-Net preserves the representation produced by the Effort backbone and refines the same patch features through two sequential stages.

The Spatial Receptive Field Fusion module is implemented by `SafeLargeSelectiveKernel` in `training/detectors/effort_detector.py`. It combines local and wider spatial responses using a 5 x 5 depthwise convolution and a dilated 7 x 7 depthwise convolution.

The Spectral Modulation Feature Refinement module is implemented by `PhaseAmpFreqAttentionBlock`. It separately modulates the amplitude and phase of the spatially refined feature and generates a frequency correction.

The complete model applies spatial refinement before frequency refinement.

## Environment

The dependencies can be installed using:

bash install.sh

The implementation uses Python, PyTorch, Transformers, OpenCV, scikit-learn, and the dependencies listed in `install.sh`.

## Pretrained Model

The visual backbone is initialized from CLIP ViT-L/14.

Download the pretrained model and modify the path used by `CLIPModel.from_pretrained` in:

training/detectors/effort_detector.py

The path should point to the local directory containing the pretrained CLIP ViT-L/14 model.

## Dataset Preparation

The experiments use FaceForensics++ as the training dataset. Celeb-DF-v1, Celeb-DF-v2, DFDCP, and FaceShifter are used for cross-domain evaluation.

Dataset preprocessing follows the DeepfakeBench protocol. The preprocessing settings are provided in:

preprocessing/config.yaml

The dataset paths should be adjusted according to the local storage environment.

The input resolution is 224 x 224 pixels. Eight frames are sampled from each video for training and testing. The FaceForensics++ c23 compression level is used for the main experiments.

## Training Configuration

The detector configuration is provided in:

training/config/detector/effort.yaml

The main training settings are:

- Training dataset: FaceForensics++
- Test datasets: Celeb-DF-v1, Celeb-DF-v2, DFDCP, and FaceShifter
- Input resolution: 224 x 224
- Training batch size: 32
- Test batch size: 32
- Number of epochs: 10
- Optimizer: Adam
- Learning rate: 0.0002
- Weight decay: 0.0005
- Loss function: cross-entropy
- Main prediction loss weight: 1.00
- Backbone prediction loss weight: 0.10
- Enhanced prediction loss weight: 0.45
- Evaluation metric: AUC

The model can be trained using:

CUDA_VISIBLE_DEVICES=0 python training/train.py --detector_path ./training/config/detector/effort.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "FaceShifter"

## Independent Runs

The reported mean and sample standard deviation are calculated from three independent runs using the following random seeds:

- 1024
- 2026
- 3028

The random seed is controlled by `manualSeed` in:

training/config/detector/effort.yaml

For each independent run, set `manualSeed` to the corresponding value before training.

## Default Model Parameters

The default parameters used by SFFR-Net are:

- SRFF dilation rate: 3
- SRFF depthwise kernel sizes: 5 x 5 and 7 x 7
- Channel calibration coefficients: alpha_c = 0.80 and beta_c = 0.20
- Spatial fusion weight: omega_s = 1.30
- Frequency fusion weight: omega_f = 0.45
- Initial spatial correction scale: lambda_s = 0.015
- Range of lambda_s: [0, 0.12]
- Initial frequency correction scale: lambda_f = 0.008
- Range of lambda_f: [0, 0.10]
- Initial frequency modulation scale: rho_f = 0.08
- Range of rho_f: [0.02, 0.15]
- Initial enhanced-logit scale: 0.35
- Range of enhanced-logit scale: [0.10, 0.65]

## Ablation Configurations

The ablation study evaluates six configurations:

- Baseline: Effort without SRFF or SMFR
- SRFF: only spatial refinement
- SMFR: only frequency refinement
- SRFF // SMFR: parallel spatial and frequency refinement
- SMFR + SRFF: frequency refinement followed by spatial refinement
- SRFF + SMFR: spatial refinement followed by frequency refinement

The symbol `//` denotes parallel processing. The symbol `+` denotes sequential processing, with the order read from left to right.

The complete SFFR-Net uses the SRFF + SMFR configuration.

## Sensitivity Settings

The default setting is:

- alpha_c = 0.80 and beta_c = 0.20
- omega_s = 1.30
- omega_f = 0.45

The sensitivity study evaluates the following alternatives:

- alpha_c = 0.70 and beta_c = 0.30
- alpha_c = 0.90 and beta_c = 0.10
- omega_f = 0.30
- omega_f = 0.60
- omega_s = 1.00
- omega_s = 1.60

All sensitivity experiments use the same training and evaluation protocol as the default model.

## Testing

A trained model can be evaluated using:

CUDA_VISIBLE_DEVICES=0 python training/test.py --detector_path ./training/config/detector/effort.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "FaceShifter" --weights_path /path/to/ckpt_best.pth

Replace `/path/to/ckpt_best.pth` with the path of the trained checkpoint.

## Acknowledgement

This implementation is developed based on DeepfakeBench and the Effort detector. We thank the authors for releasing their code.
