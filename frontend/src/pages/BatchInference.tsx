import { useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";
import type { ECharts } from "echarts";
import { Check, ChevronLeft, ChevronRight, Clipboard, Play, Trash2, Upload } from "lucide-react";
import {
  BatchInferenceResult,
  BatchSample,
  ExplanationResult,
  FeatureDef,
  LogicalModel,
  explain,
  getModelFeatures,
  inferBatch,
} from "../api/client";

const CHART_GROUP = "batch-inference";

type Row = {
  id: string;
  timestamp: string;
  features: Record<string, number | "">;
};

type Props = {
  models: LogicalModel[];
};

const nowLocal = () => {
  const d = new Date();
  d.setSeconds(0, 0);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
};

function createRow(featureDefs: FeatureDef[]): Row {
  return {
    id: crypto.randomUUID(),
    timestamp: nowLocal(),
    features: Object.fromEntries(featureDefs.map((feature) => [feature.name, feature.default])),
  };
}

export default function BatchInference({ models }: Props) {
  const [modelName, setModelName] = useState(models[0]?.name ?? "");
  const [featureDefs, setFeatureDefs] = useState<FeatureDef[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [result, setResult] = useState<BatchInferenceResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [contributorsExpanded, setContributorsExpanded] = useState(true);
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState("");

  const selectedModel = modelName || models[0]?.name || "";

  useEffect(() => {
    if (!selectedModel) {
      setFeatureDefs([]);
      setRows([]);
      setResult(null);
      return;
    }

    setError("");
    getModelFeatures(selectedModel)
      .then((defs) => {
        setFeatureDefs(defs);
        setRows([]);
        setResult(null);
      })
      .catch((err) => {
        setFeatureDefs([]);
        setRows([]);
        setResult(null);
        setError(err instanceof Error ? err.message : "Failed to load model features");
      });
  }, [selectedModel]);

  useEffect(() => {
    if (!models.some((model) => model.name === modelName)) {
      setModelName(models[0]?.name ?? "");
    }
  }, [modelName, models]);

  const axisPointerLink = useMemo(
    () => [{ group: CHART_GROUP, axis: "x" as const }],
    [],
  );

  const featureOption = useMemo(() => {
    if (!result) return {};

    return {
      axisPointer: { link: axisPointerLink },
      dataZoom: [{ type: "inside" }, { type: "slider" }],
      grid: { left: 48, right: 24, top: 58, bottom: 72 },
      legend: { top: 0, type: "scroll" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "time" },
      yAxis: { type: "value" },
      series: featureDefs.map((feature) => ({
        name: feature.name,
        type: "line",
        showSymbol: false,
        data: result.predictions.map((prediction, index) => [
          new Date(prediction.timestamp),
          rows[index]?.features[feature.name],
        ]),
      })),
    };
  }, [axisPointerLink, featureDefs, result, rows]);

  const scoreOption = useMemo(() => {
    if (!result) return {};

    const anomalyPoints = result.predictions
      .filter((prediction) => prediction.is_anomaly)
      .map((prediction) => [new Date(prediction.timestamp), prediction.anomaly_score]);

    return {
      axisPointer: { link: axisPointerLink },
      dataZoom: [{ type: "inside" }, { type: "slider" }],
      grid: { left: 48, right: 24, top: 28, bottom: 72 },
      tooltip: { trigger: "axis" },
      xAxis: { type: "time" },
      yAxis: { type: "value" },
      series: [
        {
          color: "#236d4a",
          data: result.predictions.map((prediction) => [new Date(prediction.timestamp), prediction.anomaly_score]),
          name: "Score",
          showSymbol: false,
          type: "line",
        },
        {
          data: anomalyPoints,
          itemStyle: { color: "#c0392b" },
          name: "Anomaly",
          symbolSize: 12,
          type: "scatter",
          z: 10,
        },
      ],
    };
  }, [axisPointerLink, result]);

  function onChartReady(instance: ECharts) {
    instance.group = CHART_GROUP;
    echarts.connect(CHART_GROUP);
  }

  const contributorBars = useMemo(() => {
    const values = explanation?.shap_values ?? {};
    const max = Math.max(0.001, ...Object.values(values).map((v) => Math.abs(v)));
    return featureDefs.map(({ name }) => ({
      name,
      value: values[name] ?? 0,
      width: `${(Math.abs(values[name] ?? 0) / max) * 100}%`,
    }));
  }, [explanation, featureDefs]);

  async function explainAtTimestamp(timestampIso: string) {
    if (!selectedModel || !result) return;
    const predictionIndex = result.predictions.findIndex((p) => p.timestamp === timestampIso);
    if (predictionIndex < 0) return;
    const row = rows[predictionIndex];
    if (!row) return;

    const features = Object.fromEntries(
      featureDefs.map((feature) => [feature.name, row.features[feature.name]]),
    );
    if (Object.values(features).some((value) => value === "" || typeof value !== "number" || Number.isNaN(value))) {
      return;
    }

    setSelectedTimestamp(timestampIso);
    setContributorsExpanded(true);
    setExplainBusy(true);
    setExplainError("");
    try {
      const next = await explain(selectedModel, features as Record<string, number>);
      setExplanation(next);
    } catch (err) {
      setExplanation(null);
      setExplainError(err instanceof Error ? err.message : "Explain failed");
    } finally {
      setExplainBusy(false);
    }
  }

  function onChartClick(params: { value?: unknown; data?: unknown }) {
    const data = (params.value ?? params.data) as unknown;
    if (!Array.isArray(data) || data.length === 0) return;
    const ts = new Date(data[0] as string | number | Date);
    if (Number.isNaN(ts.getTime())) return;
    void explainAtTimestamp(ts.toISOString());
  }

  const chartEvents = { click: onChartClick };

  const selectedPrediction = useMemo(() => {
    if (!result || !selectedTimestamp) return null;
    return result.predictions.find((p) => p.timestamp === selectedTimestamp) ?? null;
  }, [result, selectedTimestamp]);

  useEffect(() => {
    setSelectedTimestamp(null);
    setExplanation(null);
    setExplainError("");
  }, [result]);

  const anomalyCount = result?.predictions.filter((prediction) => prediction.is_anomaly).length ?? 0;

  function changeModel(nextModel: string) {
    if (rows.length > 0 && !window.confirm("Changing the model clears the current rows. Continue?")) {
      return;
    }
    setModelName(nextModel);
  }

  function addRow() {
    setError("");
    setResult(null);
    setRows((current) => [...current, createRow(featureDefs)]);
  }

  function removeRow(id: string) {
    setError("");
    setResult(null);
    setRows((current) => current.filter((row) => row.id !== id));
  }

  function updateTimestamp(id: string, timestamp: string) {
    setRows((current) => current.map((row) => (row.id === id ? { ...row, timestamp } : row)));
  }

  function updateFeature(id: string, name: string, value: string) {
    setRows((current) =>
      current.map((row) =>
        row.id === id
          ? { ...row, features: { ...row.features, [name]: value === "" ? "" : Number(value) } }
          : row,
      ),
    );
  }

  function importFromCsv(file: File) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result;
      if (typeof text !== "string") return;

      const lines = text.trim().split(/\r?\n/);
      if (lines.length < 2) {
        setError("CSV must have a header row and at least one data row");
        return;
      }

      const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
      const timestampIndex = headers.indexOf("timestamp");
      if (timestampIndex === -1) {
        setError("CSV must include a 'timestamp' column");
        return;
      }

      // Map each model feature to its CSV column index (-1 if absent)
      const featureIndices: Record<string, number> = {};
      for (const feature of featureDefs) {
        featureIndices[feature.name] = headers.indexOf(feature.name.toLowerCase());
      }

      const newRows: Row[] = [];
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const values = line.split(",").map((v) => v.trim());
        const rawTs = values[timestampIndex];
        const parsed = new Date(rawTs);
        if (Number.isNaN(parsed.getTime())) {
          setError(`Row ${i}: invalid timestamp "${rawTs}"`);
          return;
        }

        const localTs = new Date(parsed.getTime() - parsed.getTimezoneOffset() * 60000)
          .toISOString()
          .slice(0, 16);

        const features: Record<string, number | ""> = {};
        for (const feature of featureDefs) {
          const colIdx = featureIndices[feature.name];
          if (colIdx === -1) {
            // Column not in CSV — leave empty
            features[feature.name] = "";
          } else {
            const raw = values[colIdx];
            const num = Number(raw);
            features[feature.name] = raw === undefined || raw === "" || Number.isNaN(num) ? "" : num;
          }
        }

        newRows.push({ id: crypto.randomUUID(), timestamp: localTs, features });
      }

      if (newRows.length === 0) {
        setError("No valid rows found in CSV");
        return;
      }

      setError("");
      setResult(null);
      setRows(newRows);
    };
    reader.readAsText(file);
  }

  const curlBatch = useMemo(() => {
    if (!selectedModel || rows.length === 0) return "";
    const samples = rows.map((row) => ({
      timestamp: new Date(row.timestamp).toISOString(),
      features: Object.fromEntries(featureDefs.map((f) => [f.name, row.features[f.name]])),
    }));
    return `curl -s -X POST http://localhost:8000/api/infer/${selectedModel}/batch \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ datapoints: samples })}'`;
  }, [selectedModel, rows, featureDefs]);

  function copyCurl() {
    void navigator.clipboard.writeText(curlBatch).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  async function runBatch() {
    if (!selectedModel) return;

    const samples: BatchSample[] = [];
    for (const [index, row] of rows.entries()) {
      const parsedTimestamp = new Date(row.timestamp);
      if (!row.timestamp || Number.isNaN(parsedTimestamp.getTime())) {
        setError(`Row ${index + 1} has missing values`);
        return;
      }

      const features = Object.fromEntries(
        featureDefs.map((feature) => [feature.name, row.features[feature.name]]),
      );
      if (Object.values(features).some((value) => value === "" || typeof value !== "number" || Number.isNaN(value))) {
        setError(`Row ${index + 1} has missing values`);
        return;
      }

      samples.push({
        timestamp: parsedTimestamp.toISOString(),
        features: features as Record<string, number>,
      });
    }

    setBusy(true);
    setError("");
    try {
      setResult(await inferBatch(selectedModel, samples));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch inference failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="inference-page">
      <section className="panel form-panel">
        <header className="panel-header">
          <h1>Batch Inference</h1>
        </header>
        {error && <div className="alert">{error}</div>}
        <div className="batch-toolbar">
          <label>
            <span>Model</span>
            <select value={selectedModel} onChange={(event) => changeModel(event.target.value)}>
              {models.map((model) => <option key={model.name}>{model.name}</option>)}
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={addRow} disabled={!selectedModel || featureDefs.length === 0}>
            Add Row
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!selectedModel || featureDefs.length === 0}
          >
            <Upload size={18} />
            <span>Import CSV</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) importFromCsv(file);
              event.target.value = "";
            }}
          />
          <button className="primary-button" type="button" onClick={() => void runBatch()} disabled={rows.length === 0 || !selectedModel || busy}>
            <Play size={18} />
            <span>{busy ? "Running" : "Run"}</span>
          </button>
          <button className="secondary-button" type="button" onClick={copyCurl} disabled={!result}>
            {copied ? <Check size={18} /> : <Clipboard size={18} />}
            <span>{copied ? "Copied!" : "Copy cURL"}</span>
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="table-wrap">
          <table className="batch-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                {featureDefs.map((feature) => <th key={feature.name}>{feature.name}</th>)}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <input
                      type="datetime-local"
                      value={row.timestamp}
                      onChange={(event) => updateTimestamp(row.id, event.target.value)}
                    />
                  </td>
                  {featureDefs.map((feature) => (
                    <td key={feature.name}>
                      <input
                        type="number"
                        step="any"
                        value={row.features[feature.name] ?? ""}
                        onChange={(event) => updateFeature(row.id, feature.name, event.target.value)}
                      />
                    </td>
                  ))}
                  <td className="actions">
                    <button className="icon-button" type="button" onClick={() => removeRow(row.id)} title="Delete row">
                      <Trash2 size={17} />
                    </button>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={featureDefs.length + 2} className="empty">Click Add Row or Import CSV to start</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div
        className="batch-charts-layout"
        data-contributors-expanded={contributorsExpanded ? "true" : "false"}
      >
        <div className="batch-charts-main">
          <section className="panel chart-panel">
            <header className="panel-header">
              <h1>Features</h1>
            </header>
            {result ? (
              <ReactECharts
                option={featureOption}
                style={{ height: 360 }}
                notMerge
                lazyUpdate
                onChartReady={onChartReady}
                onEvents={chartEvents}
              />
            ) : (
              <div className="empty">Run batch inference to see feature values over time</div>
            )}
          </section>

          <section className="panel chart-panel">
            <header className="panel-header">
              <h1>Anomaly Scores</h1>
            </header>
            {result ? (
              <>
                <ReactECharts
                  option={scoreOption}
                  style={{ height: 360 }}
                  notMerge
                  lazyUpdate
                  onChartReady={onChartReady}
                  onEvents={chartEvents}
                />
                <div className="chart-caption">
                  {anomalyCount === 0
                    ? "No anomalies detected"
                    : `${anomalyCount} of ${result.predictions.length} flagged as anomalies`}
                </div>
              </>
            ) : (
              <div className="empty">Run batch inference to see anomaly scores</div>
            )}
          </section>
        </div>

        <aside className={`contributors-panel ${contributorsExpanded ? "expanded" : "collapsed"}`}>
          {contributorsExpanded ? (
            <>
              <header className="panel-header contributors-header">
                <h1>Contributors</h1>
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => setContributorsExpanded(false)}
                  title="Minimize"
                >
                  <ChevronRight size={17} />
                </button>
              </header>
              {explainError && <div className="alert">{explainError}</div>}
              {selectedPrediction ? (
                <>
                  <div className="contributors-meta">
                    <div>
                      <span>Timestamp</span>
                      <strong>{new Date(selectedPrediction.timestamp).toLocaleString()}</strong>
                    </div>
                    <div>
                      <span>Score</span>
                      <strong>{selectedPrediction.anomaly_score.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Verdict</span>
                      <strong className={selectedPrediction.is_anomaly ? "anomaly-text" : "normal-text"}>
                        {selectedPrediction.is_anomaly ? "Anomaly" : "Normal"}
                      </strong>
                    </div>
                  </div>
                  {explainBusy ? (
                    <div className="empty">Explaining…</div>
                  ) : explanation ? (
                    <div className="bars">
                      {contributorBars.map((bar) => (
                        <div className="bar-row" key={bar.name}>
                          <span>{bar.name}</span>
                          <div className="bar-track">
                            <div
                              className={bar.value < 0 ? "bar negative" : "bar positive"}
                              style={{ width: bar.width }}
                            />
                          </div>
                          <strong>{bar.value.toFixed(4)}</strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="empty">No explanation available</div>
                  )}
                </>
              ) : (
                <div className="empty">Click a point on either chart to explain it</div>
              )}
            </>
          ) : (
            <button
              className="contributors-toggle"
              type="button"
              onClick={() => setContributorsExpanded(true)}
              title="Expand contributors"
            >
              <ChevronLeft size={17} />
              <span>Contributors</span>
            </button>
          )}
        </aside>
      </div>
    </div>
  );
}
