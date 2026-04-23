import { FormEvent, useEffect, useMemo, useState } from "react";
import { Check, Clipboard, Play } from "lucide-react";
import { ExplanationResult, FeatureDef, InferenceResult, LogicalModel, explain, getModelFeatures, infer } from "../api/client";

type Props = {
  models: LogicalModel[];
};

export default function Inference({ models }: Props) {
  const [modelName, setModelName] = useState(models[0]?.name ?? "");
  const [featureDefs, setFeatureDefs] = useState<FeatureDef[]>([]);
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<InferenceResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const selectedModel = modelName || models[0]?.name || "";

  useEffect(() => {
    if (!selectedModel) return;
    setPrediction(null);
    setExplanation(null);
    getModelFeatures(selectedModel)
      .then((defs) => {
        setFeatureDefs(defs);
        setFeatures(Object.fromEntries(defs.map((f) => [f.name, f.default])));
      })
      .catch(() => {});
  }, [selectedModel]);

  const bars = useMemo(() => {
    const values = explanation?.shap_values ?? {};
    const max = Math.max(0.001, ...Object.values(values).map((v) => Math.abs(v)));
    return featureDefs.map(({ name }) => ({
      name,
      value: values[name] ?? 0,
      width: `${(Math.abs(values[name] ?? 0) / max) * 100}%`,
    }));
  }, [explanation, featureDefs]);

  const curlBody = useMemo(() => JSON.stringify({ features }, null, 2), [features]);

  const curlInfer = useMemo(
    () =>
      selectedModel
        ? `curl -s -X POST http://localhost:8000/api/infer/${selectedModel} \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ features })}'`
        : "",
    [selectedModel, features],
  );

  const curlExplain = useMemo(
    () =>
      selectedModel
        ? `curl -s -X POST http://localhost:8000/api/infer/${selectedModel}/explain \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ features })}'`
        : "",
    [selectedModel, features],
  );

  function copyCurl() {
    void navigator.clipboard.writeText(curlInfer).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!selectedModel) return;
    setBusy(true);
    setError("");
    try {
      const nextPrediction = await infer(selectedModel, features);
      setPrediction(nextPrediction);
      explain(selectedModel, features)
        .then(setExplanation)
        .catch(() => {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="inference-page">
      <div className="inference-grid">
        <form className="panel form-panel" onSubmit={(event) => void submit(event)}>
          <header className="panel-header">
            <h1>Inference</h1>
          </header>
          {error && <div className="alert">{error}</div>}
          <label>
            <span>Model</span>
            <select value={selectedModel} onChange={(event) => setModelName(event.target.value)}>
              {models.map((model) => <option key={model.name}>{model.name}</option>)}
            </select>
          </label>
          {featureDefs.map(({ name }) => (
            <label key={name}>
              <span>{name}</span>
              <input
                type="number"
                step="any"
                value={features[name] ?? ""}
                onChange={(event) => setFeatures({ ...features, [name]: Number(event.target.value) })}
              />
            </label>
          ))}
          <button className="primary-button" type="submit" disabled={busy || !selectedModel || featureDefs.length === 0}>
            <Play size={18} />
            <span>{busy ? "Running" : "Run"}</span>
          </button>
          <button className="secondary-button" type="button" onClick={copyCurl} disabled={!prediction}>
            {copied ? <Check size={18} /> : <Clipboard size={18} />}
            <span>{copied ? "Copied!" : "Copy cURL"}</span>
          </button>
        </form>

        <section className="panel result-panel">
          <header className="panel-header">
            <h1>Result</h1>
          </header>
          {prediction ? (
            <>
              <div className={prediction.is_anomaly ? "verdict anomaly" : "verdict normal"}>
                {prediction.is_anomaly ? "Anomaly" : "Normal"}
              </div>
              <div className="metric">
                <span>Raw prediction</span>
                <strong>{prediction.raw_prediction}</strong>
              </div>
              <div className="bars">
                {bars.map((bar) => (
                  <div className="bar-row" key={bar.name}>
                    <span>{bar.name}</span>
                    <div className="bar-track">
                      <div className={bar.value < 0 ? "bar negative" : "bar positive"} style={{ width: bar.width }} />
                    </div>
                    <strong>{bar.value.toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty">Run inference to see predictions and SHAP values</div>
          )}
        </section>
      </div>

      {selectedModel && (
        <section className="panel debug-panel">
          <header className="panel-header">
            <h1>Debug</h1>
          </header>
          <div className="debug-block">
            <div className="debug-label">Request body</div>
            <pre className="code-block">{curlBody}</pre>
          </div>
          <div className="debug-block">
            <div className="debug-label">cURL — infer</div>
            <pre className="code-block">{curlInfer}</pre>
          </div>
          <div className="debug-block">
            <div className="debug-label">cURL — explain</div>
            <pre className="code-block">{curlExplain}</pre>
          </div>
          {(prediction || explanation) && (
            <div className="debug-block">
              <div className="debug-label">Response — infer</div>
              <pre className="code-block">{JSON.stringify(prediction, null, 2)}</pre>
            </div>
          )}
          {explanation && (
            <div className="debug-block">
              <div className="debug-label">Response — explain</div>
              <pre className="code-block">{JSON.stringify(explanation, null, 2)}</pre>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
