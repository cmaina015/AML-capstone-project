"use client";
import { pct } from "@/lib/format";
import { SERVER_MODE } from "@/lib/mode";
import { MODEL_COMPARISON, STUDY_SPLIT } from "@/lib/study";
import type { MetricKey } from "@/lib/types";
import { useInfo } from "@/lib/useInfo";
import { Figure, Loading, Note, PageHeader, Section, Table } from "./common";

const STEPS = [
  ["data", "1. Data"],
  ["selection", "2. Model selection"],
  ["split", "3. Train/test splitting"],
  ["preprocessing", "4. Preprocessing"],
  ["training", "5. Training"],
  ["threshold", "6. Decision threshold"],
  ["evaluation", "7. Evaluation"],
  ["gradcam", "8. Grad-CAM"],
  ["deployment", "9. Deployment"],
  ["limitations", "10. Limitations"],
];

const METRICS: [MetricKey, string][] = [
  ["malignant_recall", "Malignant recall (sensitivity)"],
  ["benign_recall", "Benign recall (specificity)"],
  ["balanced_accuracy", "Balanced accuracy"],
  ["accuracy", "Accuracy"],
  ["malignant_precision", "Malignant precision"],
  ["weighted_f1", "Weighted F1"],
  ["roc_auc", "ROC-AUC"],
];

export function PipelinePage() {
  const { info, error } = useInfo();
  const ms = (m: { mean: number; std: number }) => `${m.mean.toFixed(3)} ± ${m.std.toFixed(3)}`;

  return (
    <>
      <PageHeader title="Pipeline">
        How the deployed model was selected, trained, evaluated and packaged. Numbers on this page come from the
        training notebook (<code className="text-xs">swin_tiny_train_for_deployment.ipynb</code>) unless stated
        otherwise.
      </PageHeader>

      <div className="grid gap-8 lg:grid-cols-[180px_1fr]">
        <nav className="hidden lg:block">
          <ul className="sticky top-6 space-y-1 text-sm">
            {STEPS.map(([id, label]) => (
              <li key={id}>
                <a href={`#${id}`} className="text-fg-muted hover:text-fg">
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0">
          {error && <Note tone="yellow">{error}</Note>}
          {!info ? (
            <Loading />
          ) : (
            <>
              <Section id="data" title="1. Data">
                <p>
                  The model uses the 100X magnification subset of BreakHis: {info.dataset.n_images.toLocaleString()} H&amp;E
                  images from {info.dataset.n_patients} patients ({info.dataset.classes.benign.images} benign,{" "}
                  {info.dataset.classes.malignant.images} malignant). The patient ID is read from each filename (for
                  example <code className="text-xs">SOB_B_A-14-22549AB-100-001.png</code> belongs to patient{" "}
                  <code className="text-xs">14-22549AB</code>), which makes it possible to split the data by patient.
                  See the <a href="/data/index.html" className="text-brand-fg hover:underline">Data page</a> for the full
                  exploratory analysis.
                </p>
              </Section>

              <Section id="selection" title="2. Model selection">
                <p>
                  Ten models were compared in the original study notebook: four classical classifiers on 20 handcrafted
                  colour and texture features, three pretrained CNNs and three vision transformers. Swin Tiny had the
                  best scores and was chosen for deployment.
                </p>
                <Table
                  head={["Model", "Type", "Accuracy", "Weighted F1", "ROC-AUC"]}
                  align={["left", "left", "right", "right", "right"]}
                  rows={MODEL_COMPARISON.map((m) => [
                    m.model === "Swin Tiny" ? <strong key="s">{m.model}</strong> : m.model,
                    m.family,
                    m.accuracy.toFixed(4),
                    m.f1.toFixed(4),
                    m.auc.toFixed(4),
                  ])}
                />
                <Note tone="yellow">
                  These scores come from a single image-level split ({STUDY_SPLIT.train} / {STUDY_SPLIT.val} /{" "}
                  {STUDY_SPLIT.test} images). Images from the same patient could end up in both training and test, so
                  they overestimate performance on new patients. The deployed model was therefore re-evaluated with a
                  patient-level split (step 7).
                </Note>
              </Section>

              <Section id="split" title="3. Train/test splitting">
                <p>
                  All splits are made by patient, so every image of a patient is in the same partition. Performance is
                  estimated with 5-fold <code className="text-xs">StratifiedGroupKFold</code> (groups = patient ID,
                  stratified by class). In each fold about 15% of the training patients are held out as a validation set
                  for early stopping and threshold selection.
                </p>
                <Table
                  head={["Fold", "Test patients", "Test images", "Selected threshold"]}
                  align={["left", "right", "right", "right"]}
                  rows={info.performance.headline.folds.map((f) => [
                    `Fold ${f.fold}`,
                    f.test_patients.length,
                    f.test_at_selected.n,
                    f.threshold.toFixed(3),
                  ])}
                />
                <p>
                  The deployed model was trained on {info.model.training.n_train_patients} patients (
                  {info.model.training.n_train_images.toLocaleString()} images), with{" "}
                  {info.model.threshold_selection.n_val_patients} patients ({info.model.threshold_selection.n_val_images}{" "}
                  images) kept for validation.
                </p>
              </Section>

              <Section id="preprocessing" title="4. Preprocessing">
                <p>The same preprocessing is used in training and in this app; it is read from <code className="text-xs">model_config.json</code>.</p>
                <Table
                  head={["Step", "Setting"]}
                  rows={[
                    ["Decode", "Convert to RGB (alpha channel dropped)"],
                    ["Resize", "224 × 224, bilinear, aspect ratio not preserved, no crop"],
                    ["To tensor", "Scale pixel values to [0, 1]"],
                    ["Normalize", "ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)"],
                    ["Augmentation (training only)", info.model.training.augmentation.join(", ")],
                  ]}
                />
              </Section>

              <Section id="training" title="5. Training">
                <Table
                  head={["Setting", "Value"]}
                  rows={[
                    ["Model", `${info.model.architecture} (timm), ImageNet weights, new 2-class head`],
                    ["Optimizer", `${info.model.training.optimizer}, learning rate ${info.model.training.learning_rate}, weight decay 1e-4`],
                    ["Schedule", `Cosine annealing over ${info.model.training.max_epochs} epochs`],
                    ["Batch size", String(info.model.training.batch_size)],
                    ["Loss", "Cross-entropy"],
                    ["Mixed precision", "FP16 during training (Colab T4 GPU)"],
                    ["Early stopping", "Patience 3 on validation loss; weights restored from the best epoch"],
                    ["Deployed model", `Best epoch ${info.model.training.best_epoch} of ${info.model.training.epochs_run} run`],
                  ]}
                />
              </Section>

              <Section id="threshold" title="6. Decision threshold">
                <p>
                  The model outputs P(malignant). An image is classified as malignant if P(malignant) ≥ threshold. The
                  threshold is chosen on the validation patients as the value that gives at least{" "}
                  {pct(info.model.threshold_selection.target_malignant_recall, 0)} malignant recall while keeping as
                  many benign images correct as possible.
                </p>
                <p>
                  Deployed threshold: <strong>{info.model.threshold.toFixed(3)}</strong> (validation malignant recall{" "}
                  {pct(info.model.threshold_selection.val_malignant_recall)}, benign recall{" "}
                  {pct(info.model.threshold_selection.val_benign_recall)}).
                </p>
                <Note>
                  The thresholds selected in the five folds were{" "}
                  {info.performance.headline.fold_thresholds.map((t) => t.toFixed(2)).join(", ")}. They vary a lot because
                  each validation set contains only 2–4 benign patients, so the exact threshold is uncertain.
                </Note>
              </Section>

              <Section id="evaluation" title="7. Evaluation">
                <p>Test-fold results of the 5-fold patient-level cross-validation (mean ± standard deviation over folds):</p>
                <Table
                  head={["Metric", "Selected threshold", "Threshold 0.5"]}
                  align={["left", "right", "right"]}
                  rows={METRICS.map(([k, label]) => [
                    label,
                    ms(info.performance.headline.at_selected_threshold[k]),
                    ms(info.performance.headline["at_0.5"][k]),
                  ])}
                />
                <p>Per fold, at the threshold selected in that fold:</p>
                <Table
                  head={["Fold", "Threshold", "Malignant recall", "Benign recall", "Balanced acc.", "ROC-AUC"]}
                  align={["left", "right", "right", "right", "right", "right"]}
                  rows={info.performance.headline.folds.map((f) => [
                    `Fold ${f.fold}`,
                    f.threshold.toFixed(3),
                    pct(f.test_at_selected.malignant_recall),
                    pct(f.test_at_selected.benign_recall),
                    pct(f.test_at_selected.balanced_accuracy),
                    f.test_at_selected.roc_auc.toFixed(3),
                  ])}
                />
                <div className="grid items-start gap-6 md:grid-cols-2">
                  <div>
                    <p className="mb-2">Confusion matrix, all test folds pooled:</p>
                    <Table
                      head={["", "Predicted benign", "Predicted malignant"]}
                      align={["left", "right", "right"]}
                      rows={info.performance.headline.pooled_confusion_matrix.matrix.map((row, i) => [
                        i === 0 ? "True benign" : "True malignant",
                        row[0].toLocaleString(),
                        row[1].toLocaleString(),
                      ])}
                    />
                  </div>
                  <Figure src="/figures/cv_roc_curve.png" alt="ROC curve of each cross-validation test fold" caption="ROC curve for each test fold." />
                </div>
                {info.performance.leakage_prone_baseline && (
                  <Note tone="yellow">
                    For comparison, the same training procedure on the original image-level split gives ROC-AUC{" "}
                    {info.performance.leakage_prone_baseline.test_at_selected.roc_auc.toFixed(4)}. In that split all{" "}
                    {info.performance.leakage_prone_baseline.leakage.test_patients_also_in_train} of{" "}
                    {info.performance.leakage_prone_baseline.leakage.test_patients} test patients also had images in the
                    training set, which is why its scores are higher than the patient-level results above.
                  </Note>
                )}
              </Section>

              <Section id="gradcam" title="8. Grad-CAM">
                <p>
                  Grad-CAM uses the gradients of the predicted class with respect to the output of the last Swin block (
                  <code className="text-xs">{info.explainability.target_layer}</code>). At 224 × 224 input this layer
                  is a {info.explainability.cam_grid_hw.join(" × ")} grid of 768-channel features, so the heatmap has 49
                  cells, each covering 32 × 32 pixels, and is then upsampled to the image size.
                </p>
                <ul className="list-disc space-y-1 pl-5">
                  <li>The heatmap shows regions that increased the score of the predicted class.</li>
                  <li>It is coarse and cannot point to individual cells or nuclei.</li>
                  <li>A highlighted region is not necessarily diagnostic. In some validation images the map highlighted fat tissue.</li>
                </ul>
              </Section>

              <Section id="deployment" title="9. Deployment">
                {SERVER_MODE ? (
                  <>
                    <Table
                      head={["Component", "Details"]}
                      rows={[
                        ["Model file", "The trained weights as safetensors, SHA-256 checked when the server starts"],
                        ["Where it runs", "On the server: FastAPI + PyTorch (CPU). The browser uploads the image to POST /api/predict; images are not stored"],
                        ["Grad-CAM", "pytorch-grad-cam on the output of the last Swin block, computed per request"],
                        ["Preprocessing", "The same torchvision transform as in training, rebuilt from model_config.json"],
                        ["Frontend", "Next.js, exported as static files and served by FastAPI"],
                        ["Hosting", "Hugging Face Docker Space (one container)"],
                      ]}
                    />
                    <p className="mt-3 text-sm text-fg-2">
                      The API is documented interactively at{" "}
                      <a href="/docs" className="text-brand-fg hover:underline">/docs</a>, where an image can be
                      uploaded to the prediction endpoint directly. A static version of the same app runs the model
                      in the browser instead (ONNX Runtime Web) and was checked against this server: the largest
                      probability difference was {info.browser_model.parity.prob.toExponential(0)}.
                    </p>
                  </>
                ) : (
                <>
                <Table
                  head={["Component", "Details"]}
                  rows={[
                    ["Model file", `ONNX export of the trained weights (${(info.browser_model.bytes / 1024 / 1024).toFixed(0)} MB), SHA-256 checked after download and cached by the browser`],
                    ["Where it runs", "In the visitor's browser with ONNX Runtime Web (WebAssembly). Images are never uploaded."],
                    ["Grad-CAM", "The gradient through the final LayerNorm, pooling and linear layer is written out in closed form inside the ONNX graph, because browsers cannot back-propagate"],
                    ["Preprocessing", "Pillow's bilinear resize ported to TypeScript, so the model gets the same pixels as in training"],
                    ["Frontend", "Next.js, exported as static files"],
                    ["Hosting", "Hugging Face static Space (no server)"],
                  ]}
                />
                <p className="mt-3 text-sm text-fg-2">
                  The browser model was checked against the PyTorch model and pytorch-grad-cam before release: the
                  largest probability difference was {info.browser_model.parity.prob.toExponential(0)} and the largest
                  Grad-CAM difference {info.browser_model.parity.cam_224.toExponential(0)} (on a 0–1 scale). A FastAPI
                  version of the same pipeline is kept in the repository for local use.
                </p>
                </>
                )}
              </Section>

              <Section id="limitations" title="10. Limitations">
                <ul className="list-disc space-y-1.5 pl-5">
                  {info.limitations.map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </Section>
            </>
          )}
        </div>
      </div>
    </>
  );
}
