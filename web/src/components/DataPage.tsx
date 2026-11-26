"use client";
import { pct } from "@/lib/format";
import { MAGNIFICATION_COUNTS } from "@/lib/study";
import { useInfo } from "@/lib/useInfo";
import { Figure, Loading, Note, PageHeader, Section, Table } from "./common";

export function DataPage() {
  const { info, error } = useInfo();
  const ds = info?.dataset;
  const totalAll = MAGNIFICATION_COUNTS.reduce((s, m) => s + m.benign + m.malignant, 0);
  const rgb = (v: number[]) => v.map((x) => x.toFixed(3)).join(", ");

  return (
    <>
      <PageHeader title="Data">
        Exploratory analysis of the BreakHis dataset used to train the model. Sample images are not shown here; they
        can be viewed on the{" "}
        <a
          className="text-brand-fg hover:underline"
          href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
          target="_blank"
          rel="noopener"
        >
          BreakHis website
        </a>
        .
      </PageHeader>

      {error && <Note tone="yellow">{error}</Note>}

      <Section title="All magnifications">
        <p>
          The dataset folder contains {totalAll.toLocaleString()} images at four magnifications. Malignant images
          outnumber benign ones at every magnification by roughly 2 to 1. Only the 100X images are used for the model, so
          that all training images have the same level of detail.
        </p>
        <div className="grid items-start gap-6 md:grid-cols-2">
          <Table
            head={["Magnification", "Benign", "Malignant", "Total", "Malignant share"]}
            align={["left", "right", "right", "right", "right"]}
            rows={MAGNIFICATION_COUNTS.map((m) => [
              m.magnification === "100X" ? <strong key="m">100X (used)</strong> : m.magnification,
              m.benign.toLocaleString(),
              m.malignant.toLocaleString(),
              (m.benign + m.malignant).toLocaleString(),
              pct(m.malignant / (m.benign + m.malignant)),
            ])}
          />
          <Figure src="/eda/counts_by_magnification.png" alt="Bar chart of image counts by magnification and class" caption="Image counts by magnification and class (study notebook)." />
        </div>
      </Section>

      {!info || !ds ? (
        <Loading />
      ) : (
        <>
          <Section title="100X subset: classes and patients">
            <p>
              The 100X subset has {ds.n_images.toLocaleString()} images from {ds.n_patients} patients. A model that always
              predicted &ldquo;malignant&rdquo; would already be {pct(ds.classes.malignant.image_fraction)} accurate, which
              is why recall per class and ROC-AUC are reported instead of accuracy alone.
            </p>
            <div className="grid items-start gap-6 md:grid-cols-2">
              <Table
                head={["Class", "Images", "Share of images", "Patients", "Images per patient"]}
                align={["left", "right", "right", "right", "right"]}
                rows={(["benign", "malignant"] as const).map((c) => [
                  c === "benign" ? "Benign" : "Malignant",
                  ds.classes[c].images.toLocaleString(),
                  pct(ds.classes[c].image_fraction),
                  ds.classes[c].patients,
                  (ds.classes[c].images / ds.classes[c].patients).toFixed(1),
                ])}
              />
              {info.figures.includes("class_distribution.png") && (
                <Figure src="/figures/class_distribution.png" alt="Images and patients per class" caption="Images and patients per class at 100X." />
              )}
            </div>
          </Section>

          <Section title="Tumour subtypes">
            <p>
              Each class contains four subtypes. Ductal carcinoma makes up{" "}
              {pct((ds.subtypes.find((s) => s.code === "DC")?.images ?? 0) / ds.n_images, 0)} of all 100X images, while
              some subtypes come from very few patients (phyllodes tumour: 3 patients).
            </p>
            <Table
              head={["Class", "Subtype", "Code", "Images", "Patients"]}
              align={["left", "left", "left", "right", "right"]}
              rows={ds.subtypes.map((s) => [
                s.class === "benign" ? "Benign" : "Malignant",
                s.name.charAt(0).toUpperCase() + s.name.slice(1),
                s.code,
                s.images,
                s.patients,
              ])}
            />
          </Section>

          <Section title="Image properties">
            <p>
              All {ds.n_images.toLocaleString()} images at 100X are {ds.image_sizes.map((s) => `${s.width} × ${s.height}`).join(", ")}{" "}
              pixels and stored as {Object.keys(ds.colour_modes).join(", ")}. In a sample of 200 images (study notebook),
              benign images were slightly brighter on average than malignant ones (median about 0.70 vs 0.68), and the
              colour channel distributions overlap a lot between the classes. Colour and brightness alone are not enough
              to separate them.
            </p>
            <div className="grid items-start gap-6 md:grid-cols-[1fr_2fr]">
              <Figure src="/eda/brightness_by_class.png" alt="Box plot of mean brightness by class" caption="Mean brightness by class (sample of 200)." />
              <Figure src="/eda/colour_channels.png" alt="Density plots of mean red, green and blue values by class" caption="Mean R, G and B values per image by class (sample of 200)." />
            </div>
            <p>Mean and standard deviation of the pixel values over all 100X images (scale 0–1):</p>
            <Table
              head={["Images", "Mean (R, G, B)", "Std (R, G, B)"]}
              rows={(["all", "benign", "malignant"] as const).map((k) => [
                k === "all" ? "All" : k === "benign" ? "Benign" : "Malignant",
                rgb(ds.rgb_0_1[k].mean),
                rgb(ds.rgb_0_1[k].std),
              ])}
            />
            <p className="text-xs text-fg-muted">
              These statistics are descriptive only. The model normalizes inputs with the ImageNet mean and standard
              deviation, as expected by the pretrained weights.
            </p>
          </Section>

          <Section title="Data checks">
            <Table
              head={["Check", "Result"]}
              rows={[
                ["Filenames with a readable patient ID", `${ds.n_images - ds.integrity.unparsed_filenames} of ${ds.n_images}`],
                ["Patients with images in both classes", String(ds.integrity.patients_in_both_classes)],
                [
                  "Patients with more than one subtype",
                  Object.entries(ds.integrity.patients_spanning_subtypes)
                    .map(([p, s]) => `${p} (${s.join(", ")})`)
                    .join("; ") || "None",
                ],
                ["Images with an exact duplicate", String(ds.integrity.identical_content_images)],
                ["Duplicates shared between different patients", String(ds.integrity.identical_content_groups_across_patients)],
              ]}
            />
            <p>
              Because duplicates only occur within the same patient, splitting by patient keeps every copy of an image
              in the same partition.
            </p>
          </Section>

          <Section title="Split used for the deployed model">
            <Table
              head={["Class", "Train patients", "Train images", "Validation patients", "Validation images"]}
              align={["left", "right", "right", "right", "right"]}
              rows={(["benign", "malignant"] as const).map((c) => {
                const s = ds.splits.deployment_model[c];
                return [c === "benign" ? "Benign" : "Malignant", s.train_patients, s.train_images.toLocaleString(), s.val_patients, s.val_images];
              })}
            />
          </Section>
        </>
      )}
    </>
  );
}
