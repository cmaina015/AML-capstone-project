export type ClassName = "benign" | "malignant";

export type MetricKey =
  | "malignant_recall"
  | "benign_recall"
  | "balanced_accuracy"
  | "accuracy"
  | "malignant_precision"
  | "weighted_f1"
  | "roc_auc";

export type MeanStd = { mean: number; std: number };
export type MetricBlock = Record<MetricKey, MeanStd>;
export type FoldMetrics = Record<MetricKey, number> & {
  threshold: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  n: number;
};

export interface Fold {
  fold: number;
  test_patients: string[];
  threshold: number;
  test_at_selected: FoldMetrics;
  "test_at_0.5": FoldMetrics;
}

export interface Subtype {
  class: ClassName;
  code: string;
  name: string;
  images: number;
  patients: number;
}

export interface Info {
  intended_use: string;
  limits: { max_upload_mb: number; formats: string[] };
  model: {
    architecture: string;
    weights_sha256: string;
    classes: Record<string, ClassName>;
    positive_class: ClassName;
    threshold: number;
    reference_threshold: number;
    decision_rule: string;
    threshold_selection: {
      criterion: string;
      target_malignant_recall: number;
      val_malignant_recall: number;
      val_benign_recall: number;
      n_val_patients: number;
      n_val_images: number;
      cv_fold_thresholds: number[];
    };
    preprocessing: { resize: string; normalize: string; details: { normalize?: { mean: number[]; std: number[] } } };
    training: {
      n_train_patients: number;
      n_train_images: number;
      best_epoch: number;
      epochs_run: number;
      max_epochs: number;
      augmentation: string[];
      optimizer: string;
      learning_rate: number;
      batch_size: number;
    };
    versions: Record<string, string | null>;
  };
  dataset: {
    dataset: string;
    dataset_url: string;
    magnification: string;
    stain: string;
    n_images: number;
    n_patients: number;
    classes: Record<ClassName, { images: number; patients: number; image_fraction: number }>;
    subtypes: Subtype[];
    image_sizes: { width: number; height: number; count: number }[];
    colour_modes: Record<string, number>;
    rgb_0_1: Record<"all" | ClassName, { mean: number[]; std: number[] }>;
    integrity: {
      unparsed_filenames: number;
      patients_in_both_classes: number;
      patients_spanning_subtypes: Record<string, string[]>;
      identical_content_images: number;
      identical_content_groups_across_patients: number;
    };
    splits: {
      deployment_model: Record<
        ClassName,
        { train_patients: number; train_images: number; val_patients: number; val_images: number }
      >;
    };
  };
  performance: {
    headline: {
      label: string;
      n_folds: number;
      at_selected_threshold: MetricBlock;
      "at_0.5": MetricBlock;
      fold_thresholds: number[];
      pooled_confusion_matrix: { labels: string[]; matrix: number[][] };
      folds: Fold[];
    };
    leakage_prone_baseline: null | {
      label: string;
      leakage: {
        test_patients: number;
        test_patients_also_in_train: number;
        test_images_from_leaked_patients: number;
        test_images: number;
      };
      test_at_selected: Record<MetricKey, number>;
    };
  };
  explainability: { target_layer: string; cam_grid_hw: [number, number]; note: string };
  limitations: string[];
  figures: string[];
  /** ONNX export of the same weights, run in the browser (scripts/export_onnx.py). */
  browser_model: { url: string; sha256: string; bytes: number; parity: Record<string, number> };
  /** Constants app/inference.py uses, so the browser applies the same rules. */
  decision: {
    near_threshold_margin: number;
    aspect_tolerance: number;
    max_pixels: number;
    overlay_image_weight: number;
    size_hw: [number, number];
  };
}

export interface Prediction {
  label: ClassName;
  probabilities: Record<ClassName, number>;
  p_malignant: number;
  threshold: number;
  decision_rule: string;
  input: { format: string; mode: string; width: number; height: number; aspect_ratio: number; frames: number };
  warnings: string[];
  images: { model_input: string; overlay?: string; heatmap?: string };
  explanation: null | {
    method: string;
    target_class: ClassName;
    target_layer: string;
    grid_hw: [number, number];
    grid: number[][];
    flat: boolean;
    note: string;
  };
  timing_ms: { preprocess: number; predict: number; explain: number | null; total: number };
}
