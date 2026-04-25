# Tea-leaf-disease-classification
Tea leaf disease classification using ResNet transfer learning — identifying plant diseases from leaf images to support smarter, data-driven farming.


# Tea Leaf Disease Classification — Setup & Run Guide

## 1. Install Dependencies
```bash
pip install torch torchvision scikit-learn seaborn matplotlib notebook
```

## 2. Dataset Folder Structure
```
assignment02/
├── dataset/
│   ├── <disease_class_1>/
│   ├── <disease_class_2>/
│   └── <disease_class_3>/
├── vgg_model.ipynb
└── resnet_model.ipynb
```

## 3. Download Dataset from Google Drive
```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1-1qur6uv1ZcmgCiBJu9tSNDxgmyp5jtx" -O ./dataset
```

## 4. Run Notebooks
```bash
jupyter notebook
```
- Open `vgg_model.ipynb` → Run All
- Open `resnet_model.ipynb` → Run All

## 5. Outputs Generated
- `vgg16_training_curves.png` / `resnet50_training_curves.png`
- `vgg16_confusion_matrix.png` / `resnet50_confusion_matrix.png`
- `vgg16_tea_disease.pth` / `resnet50_tea_disease.pth`
- Classification report printed in notebook

## 6. Notes
- CPU: ~10–20 min per model | GPU: ~2–3 min per model
- DATA_DIR at top of each notebook — update path if needed
