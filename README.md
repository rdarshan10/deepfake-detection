# DeepFake Detection Project

This repository contains code for training and evaluating various deep learning models for deepfake detection using multiple datasets.

## Datasets

The project uses three main datasets:

1. [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
   - Size: 4.04 GB
   - Contains 140,000 images of real and fake faces
   - High-quality synthetic faces generated using StyleGAN

2. [Celeb-DF-New](https://www.kaggle.com/datasets/mogpgo/celeb-df-new)
   - Size: 207.15 MB
   - Celebrity deepfake videos and frames
   - High-quality manipulated facial content

3. [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
   - Size: 1.18 GB
   - Collection of real and computer-generated faces
   - Diverse facial expressions and angles

## Dataset Strategy

### Data Split
The datasets are split using the following ratios:
- Training Set: 60%
- Validation Set: 20%
- Test Set: 20%

This split ensures:
- Sufficient data for model training
- Adequate validation for hyperparameter tuning
- Representative test set for final evaluation

### Combined Dataset Statistics
Total Data Size: 5.42 GB
- Combined Training Set: ~3.25 GB
- Combined Validation Set: ~1.08 GB
- Combined Test Set: ~1.08 GB

## Project Structure

```
main/
│
├── README.md
├── harencing-model.ipynb        # Main notebook with baseline models
├── deepfake1092112.ipynb       # LFAT model implementation 
├── comp.ipynb                  # Comparative analysis notebook
└── deepfakemulti1111.ipynb    # Multi-dataset training notebook
```

## Models Implemented

1. CNN (Custom Convolutional Neural Network)
2. ResNet50 
3. MobileNetV2
4. VGG16
5. LFAT (Lightweight Frequency Attention Transformer)

## Proposed Model: LFAT (Lightweight Frequency Attention Transformer)

Our novel LFAT model is designed for efficient and accurate deepfake detection by combining:

1. **Frequency Domain Analysis**: Captures manipulation artifacts in frequency space
2. **Attention Mechanism**: Focuses on relevant frequency patterns
3. **Lightweight Architecture**: Optimized for real-world deployment

### Architecture Benefits
- **Efficient Resource Usage**: Only 5.5M total parameters (vs 23.8M in ResNet)
- **Compact Model Size**: 21.19 MB (vs 90.98 MB ResNet, 57.64 MB VGG16)
- **Balanced Parameters**: 3.29M trainable vs 2.25M non-trainable parameters
- **Memory Efficient**: 3.8x smaller than ResNet50, 2.7x smaller than VGG16

### Performance Benchmarks

#### Model Size and Parameters
| Model     | Total Parameters | Trainable Params | Non-Trainable | Size (MB) |
|-----------|------------------|------------------|---------------|-----------|
| LFAT      | 5,555,777       | 3,297,793       | 2,257,984     | 21.19    |
| ResNet50  | 23,850,113      | 262,401         | 23,587,712    | 90.98    |
| VGG16     | 15,108,929      | 7,473,665       | 7,635,264     | 57.64    |
| MobileNet | 3,360,193       | 131,329         | 3,228,864     | 12.82    |
| CNN       | 14,839,105      | 14,839,105      | 0             | 56.61    |

#### Inference Performance
| Model     | Load Time (s) | Predict Time (s) | Total Time (s) |
|-----------|---------------|------------------|----------------|
| LFAT      | 1.5551       | 0.8948          | 2.4499        |
| ResNet50  | 1.5590       | 1.2582          | 2.8172        |
| VGG16     | 0.4124       | 0.2489          | 0.6613        |
| MobileNet | 0.6003       | 0.3375          | 0.9378        |
| CNN       | 0.3673       | 0.1036          | 0.4709        |

### Key Advantages
1. **Balanced Performance**: While maintaining competitive inference speed
2. **Resource Efficiency**: Significantly smaller model size than traditional architectures
3. **Effective Feature Learning**: Through frequency-domain analysis
4. **Deployment Ready**: Optimized for real-world applications

## Key Features

- Data preprocessing and augmentation
- Multi-model training and evaluation
- Transfer learning with pretrained models
- Frequency domain analysis with LFAT
- Model performance visualization
- Real-time inference capabilities

## Training Pipeline

1. Dataset Splitting (60% train, 20% validation, 20% test)
2. Data Augmentation:
   - Rotation
   - Width/height shifts
   - Horizontal flips
   - Zoom
   - Normalization

## Model Performance Metrics

Each model is evaluated using:
- Accuracy
- F1 Score
- Precision
- Recall
- AUC-ROC
- Confusion Matrix

## Usage

1. Install Requirements:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

2. Set up directories for datasets:
```python
base_dir = 'd:/deepfake/Celeb-DF-New'
output_dir = 'd:/deepfake/split_data'
```

3. Training a model:
```python
python comp.ipynb  # For comparative analysis
python deepfakemulti1111.ipynb  # For multi-dataset training
```

## Real-time Inference

The project includes real-time inference capabilities for detecting deepfakes in images:

```python
img_path = "path/to/image.jpg"
model_path = "path/to/saved_model.h5"
make_single_image_prediction(img_path, model_path, model_name)
```

## Performance Comparison

Models are compared based on:
- Training time
- Inference latency
- Memory usage
- Detection accuracy

## Contributing

Feel free to contribute by:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is open-source and available under the MIT License.
