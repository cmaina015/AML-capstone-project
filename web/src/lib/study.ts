/* Results copied from the comparative study notebook
   (breast_cancer_classification_academic_markdown.ipynb). These used a single image-level
   70/15/15 split (seed 42), so images from the same patient could fall in both train and test. */

export const MAGNIFICATION_COUNTS = [
  { magnification: "40X", benign: 641, malignant: 1370 },
  { magnification: "100X", benign: 644, malignant: 1437 },
  { magnification: "200X", benign: 623, malignant: 1390 },
  { magnification: "400X", benign: 588, malignant: 1232 },
];

export const STUDY_SPLIT = { train: 1456, val: 312, test: 313 };

export const MODEL_COMPARISON = [
  { model: "Swin Tiny", family: "Vision Transformer", accuracy: 0.9936, f1: 0.9936, auc: 0.9998 },
  { model: "DeiT Small", family: "Vision Transformer", accuracy: 0.9808, f1: 0.9809, auc: 0.9989 },
  { model: "ViT Base", family: "Vision Transformer", accuracy: 0.9521, f1: 0.9525, auc: 0.9831 },
  { model: "EfficientNetB0", family: "Pretrained CNN", accuracy: 0.9425, f1: 0.9433, auc: 0.989 },
  { model: "XGBoost", family: "Classical ML (20 features)", accuracy: 0.9393, f1: 0.9392, auc: 0.9777 },
  { model: "ResNet50", family: "Pretrained CNN", accuracy: 0.9297, f1: 0.9295, auc: 0.9826 },
  { model: "Random Forest", family: "Classical ML (20 features)", accuracy: 0.9233, f1: 0.9215, auc: 0.9763 },
  { model: "SVM", family: "Classical ML (20 features)", accuracy: 0.9137, f1: 0.9151, auc: 0.9758 },
  { model: "MobileNetV2", family: "Pretrained CNN", accuracy: 0.9137, f1: 0.9149, auc: 0.9756 },
  { model: "Logistic Regression", family: "Classical ML (20 features)", accuracy: 0.8466, f1: 0.8495, auc: 0.9088 },
];
