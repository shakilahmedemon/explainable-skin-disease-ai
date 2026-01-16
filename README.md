# Explainable Skin Disease AI

This project implements an AI system for skin disease classification with explainability features. It uses Vision Transformers (ViT) and ensemble methods to classify skin lesions with uncertainty quantification and provides explanations for its predictions.

## Features

- **Multi-model Architecture**: Uses both Vision Transformers and ensemble methods for improved accuracy
- **Explainability**: Provides GradCAM, SHAP, and other explanation methods to highlight important regions
- **Uncertainty Quantification**: Estimates prediction uncertainty using ensemble methods
- **Medical Insights**: Provides medical context for each prediction
- **Safety Warnings**: Highlights high-risk conditions requiring immediate attention
- **Streamlit Interface**: Easy-to-use web interface for uploading images and viewing results

## Architecture

The project is organized as follows:

```
├── app/
│   ├── app.py              # Main Streamlit application
│   ├── explainability.py   # Explanation methods (GradCAM, SHAP, etc.)
│   ├── medical_knowledge.py # Medical insights and information
│   ├── safety.py           # Safety checks and warnings
│   └── utils.py            # Utility functions
├── model/
│   ├── vit_model.py        # Vision Transformer implementation
│   └── ensemble/
│       └── ensemble_uncertainty.py # Ensemble model with uncertainty estimation
├── training/
│   ├── train_single_model.py   # Training script for single model
│   ├── train_ensemble.py       # Training script for ensemble model
│   ├── evaluate.py             # Evaluation utilities
│   └── calibration.py          # Model calibration methods
├── data/                     # Data directory (to be populated)
├── results/                  # Training results and logs
├── requirements.txt          # Python dependencies
├── train.py                  # Main training script
└── setup_env.py              # Environment setup script
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the setup script: `python setup_env.py`
4. Install additional packages: `pip install shap lime scipy seaborn`

## Usage

### Running the Application

```bash
streamlit run app/app.py
```

### Training New Models

```bash
python train.py
```

### Using the Demo Version

```bash
streamlit run demo_app.py
```

## Model Details

### Vision Transformer (ViT)
- Base ViT model (vit_base_patch16_224) pre-trained on ImageNet
- Custom classifier head for 7 skin disease classes
- Dropout for regularization

### Ensemble Model
- Combines ViT, ResNet50, and EfficientNetB0
- Provides uncertainty estimates through disagreement between models
- Averaged predictions for improved robustness

## Classes

The model currently supports 7 common skin conditions:
1. Melanoma
2. Basal Cell Carcinoma
3. Squamous Cell Carcinoma
4. Nevus (Mole)
5. Dermatofibroma
6. Vascular Lesion
7. Actinic Keratosis

## Explainability Methods

1. **GradCAM**: Highlights important regions in the image
2. **SHAP**: Provides pixel-level attribution
3. **Saliency Maps**: Shows gradient-based importance

## Safety Considerations

- High-risk conditions trigger immediate warnings
- Confidence thresholds alert users to uncertain predictions
- Clear disclaimers about medical use limitations
- Uncertainty quantification for risk assessment

## Data Requirements

To train the model, you need a dataset organized as:
```
data/
├── train/
│   ├── melanoma/
│   ├── basal_cell_carcinoma/
│   ├── squamous_cell_carcinoma/
│   ├── nevus/
│   ├── dermatofibroma/
│   ├── vascular_lesion/
│   └── actinic_keratosis/
├── val/
└── test/
```

## Performance Optimization

- Mixed precision training for faster computation
- Data augmentation to improve generalization
- Model calibration to improve confidence estimates
- Efficient preprocessing pipelines

## Ethical Considerations

- Fairness across different skin tones
- Privacy preservation (no data transmission)
- Transparency in model predictions
- Clear limitations and disclaimers

## Future Enhancements

- Integration with medical records systems
- Real-time camera integration
- Multi-modal inputs (clinical notes + images)
- Federated learning for privacy-preserving updates
- Mobile deployment options

## Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis and should not replace professional medical advice. Always consult with qualified healthcare providers for medical decisions.
