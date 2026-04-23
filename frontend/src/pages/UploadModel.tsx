import { FormEvent, useState } from "react";
import { FolderOpen, UploadCloud, X } from "lucide-react";
import { FeatureDef, LogicalModel, uploadModel } from "../api/client";

type Props = {
  onUploaded: (models: LogicalModel[]) => void;
};

function randomDefault() {
  return parseFloat((Math.random() * 99.9 + 0.1).toFixed(2));
}

function FileBadge({ file, onClear }: { file: File; onClear: () => void }) {
  return (
    <div className="file-badge">
      <span>{file.name}</span>
      <button type="button" className="icon-button file-badge-clear" onClick={onClear} title="Clear">
        <X size={12} />
      </button>
    </div>
  );
}

export default function UploadModel({ onUploaded }: Props) {
  const [modelName, setModelName] = useState("");
  const [predictorJoblib, setPredictorJoblib] = useState<File | null>(null);
  const [predictorSettings, setPredictorSettings] = useState<File | null>(null);
  const [explainerJoblib, setExplainerJoblib] = useState<File | null>(null);
  const [explainerSettings, setExplainerSettings] = useState<File | null>(null);
  const [featureDefs, setFeatureDefs] = useState<FeatureDef[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleDirectory(files: FileList | null) {
    if (!files || files.length === 0) return;

    const fileMap = new Map<string, File>();
    let metadataFile: File | null = null;

    for (const file of Array.from(files)) {
      // Strip the leading directory component: "iforest_v1/predictor/foo.joblib" → "predictor/foo.joblib"
      const rel = file.webkitRelativePath.replace(/^[^/]+\//, "");
      fileMap.set(rel, file);
      if (rel === "metadata.json") metadataFile = file;
    }

    if (!metadataFile) return;

    try {
      const parsed = JSON.parse(await metadataFile.text()) as Record<string, unknown>;

      if (typeof parsed.model_name === "string" && parsed.model_name) {
        setModelName(parsed.model_name);
      }

      const features = parsed.features as string[] | undefined;
      const sampleInput = parsed.sample_input as Record<string, number> | undefined;
      if (Array.isArray(features) && features.length > 0) {
        setFeatureDefs(
          features.map((name) => ({
            name,
            default:
              sampleInput?.[name] !== undefined
                ? parseFloat(sampleInput[name].toFixed(4))
                : randomDefault(),
          }))
        );
      }

      const artifacts = parsed.artifacts as Record<string, Record<string, string>> | undefined;
      if (artifacts) {
        const predJoblib = fileMap.get(artifacts.predictor?.model ?? "");
        const predSettings = fileMap.get(artifacts.predictor?.model_settings ?? "");
        const explJoblib = fileMap.get(artifacts.explainer?.model ?? "");
        const explSettings = fileMap.get(artifacts.explainer?.model_settings ?? "");

        if (predJoblib) setPredictorJoblib(predJoblib);
        if (predSettings) setPredictorSettings(predSettings);
        if (explJoblib) setExplainerJoblib(explJoblib);
        if (explSettings) setExplainerSettings(explSettings);
      }
    } catch {
      // ignore parse errors
    }
  }

  async function handlePredictorSettingsFile(file: File | null) {
    setPredictorSettings(file);
    if (featureDefs.length > 0 || !file) return;
    try {
      const parsed = JSON.parse(await file.text()) as Record<string, unknown>;
      const shape = (parsed?.inputs as Array<{ shape: number[] }>)?.[0]?.shape ?? [];
      const count = typeof shape[shape.length - 1] === "number" ? shape[shape.length - 1] : 0;
      if (count > 0) {
        setFeatureDefs(
          Array.from({ length: count }, (_, i) => ({ name: `tag${i + 1}`, default: randomDefault() }))
        );
      }
    } catch {
      // ignore parse errors
    }
  }

  function updateFeature(index: number, field: keyof FeatureDef, value: string | number) {
    setFeatureDefs((prev) => prev.map((f, i) => (i === index ? { ...f, [field]: value } : f)));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!predictorJoblib || !predictorSettings || !explainerJoblib || !explainerSettings) {
      setError("Select all four files for predictor and explainer");
      return;
    }
    setBusy(true);
    setError("");
    try {
      onUploaded(
        await uploadModel(
          modelName,
          predictorJoblib,
          predictorSettings,
          explainerJoblib,
          explainerSettings,
          featureDefs.length > 0 ? featureDefs : undefined
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="form-panel" onSubmit={(event) => void submit(event)}>
      <div className="form-hero">
        <span className="form-title">UPLOAD</span>
        <span className="form-subtitle">Register a new model artifact</span>
      </div>
      {error && <div className="alert">{error}</div>}
      <fieldset>
        <legend>
          <FolderOpen size={14} style={{ verticalAlign: "middle", marginRight: 4 }} />
          Import model directory{" "}
          <span style={{ fontWeight: 400, fontSize: "12px" }}>
            (optional — reads metadata.json and auto-fills all fields)
          </span>
        </legend>
        <label>
          <span>Model folder</span>
          {/* webkitdirectory is non-standard but broadly supported */}
          <input
            type="file"
            {...{ webkitdirectory: "" }}
            onChange={(event) => void handleDirectory(event.target.files)}
          />
        </label>
      </fieldset>
      <label>
        <span>Model name</span>
        <input
          value={modelName}
          onChange={(event) => setModelName(event.target.value)}
          placeholder="iforest_v2"
          required
        />
      </label>
      <fieldset>
        <legend>Predictor</legend>
        <label>
          <span>Joblib file</span>
          {predictorJoblib ? (
            <FileBadge file={predictorJoblib} onClear={() => setPredictorJoblib(null)} />
          ) : (
            <input
              type="file"
              accept=".joblib"
              onChange={(event) => setPredictorJoblib(event.target.files?.[0] ?? null)}
            />
          )}
        </label>
        <label>
          <span>model-settings.json</span>
          {predictorSettings ? (
            <FileBadge file={predictorSettings} onClear={() => setPredictorSettings(null)} />
          ) : (
            <input
              type="file"
              accept=".json"
              onChange={(event) => void handlePredictorSettingsFile(event.target.files?.[0] ?? null)}
            />
          )}
        </label>
      </fieldset>
      <fieldset>
        <legend>Explainer</legend>
        <label>
          <span>Joblib file</span>
          {explainerJoblib ? (
            <FileBadge file={explainerJoblib} onClear={() => setExplainerJoblib(null)} />
          ) : (
            <input
              type="file"
              accept=".joblib"
              onChange={(event) => setExplainerJoblib(event.target.files?.[0] ?? null)}
            />
          )}
        </label>
        <label>
          <span>model-settings.json</span>
          {explainerSettings ? (
            <FileBadge file={explainerSettings} onClear={() => setExplainerSettings(null)} />
          ) : (
            <input
              type="file"
              accept=".json"
              onChange={(event) => setExplainerSettings(event.target.files?.[0] ?? null)}
            />
          )}
        </label>
      </fieldset>
      {featureDefs.length > 0 && (
        <fieldset>
          <legend>
            Input Features{" "}
            <span style={{ fontWeight: 400, fontSize: "12px" }}>
              (optional — edit names and default inference values)
            </span>
          </legend>
          <div className="feature-header">
            <span>Feature name</span>
            <span>Default value</span>
          </div>
          {featureDefs.map((f, i) => (
            <div key={i} className="feature-row">
              <input
                value={f.name}
                onChange={(e) => updateFeature(i, "name", e.target.value)}
                placeholder={`tag${i + 1}`}
              />
              <input
                type="number"
                step="0.01"
                value={f.default}
                onChange={(e) => updateFeature(i, "default", parseFloat(e.target.value) || 0.1)}
              />
            </div>
          ))}
        </fieldset>
      )}
      <button className="primary-button" type="submit" disabled={busy}>
        <UploadCloud size={18} />
        <span>{busy ? "Uploading" : "Upload"}</span>
      </button>
    </form>
  );
}
