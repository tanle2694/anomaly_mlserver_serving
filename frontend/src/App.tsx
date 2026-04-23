import { useEffect, useMemo, useState } from "react";
import { Database, LineChart } from "lucide-react";
import { LogicalModel, listModels } from "./api/client";
import ModelsList from "./pages/ModelsList";
import UploadModel from "./pages/UploadModel";
import BatchInference from "./pages/BatchInference";
import UploadModal from "./components/UploadModal";

type View = "models" | "inference";

const views: Array<{ id: View; label: string; icon: typeof Database }> = [
  { id: "models", label: "Models", icon: Database },
  { id: "inference", label: "Inference", icon: LineChart },
];

export default function App() {
  const [view, setView] = useState<View>("models");
  const [models, setModels] = useState<LogicalModel[]>([]);
  const [error, setError] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);

  async function refreshModels() {
    setError("");
    try {
      setModels(await listModels());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
    }
  }

  useEffect(() => {
    void refreshModels();
  }, []);

  const predictorModels = useMemo(
    () => models.filter((model) => model.has_predictor && model.predictor_loaded),
    [models],
  );

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-title">ANOMALY</span>
          <span className="brand-sub">Detection Lab</span>
        </div>
        <nav className="nav">
          {views.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={view === item.id ? "nav-button active" : "nav-button"}
                onClick={() => setView(item.id)}
                type="button"
              >
                <Icon size={15} />
                <span className="nav-label">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>

      <section className="workspace">
        {error && <div className="alert">{error}</div>}
        {view === "models" && (
          <ModelsList
            models={models}
            onModelsChanged={setModels}
            onUpload={() => setUploadOpen(true)}
          />
        )}
        {view === "inference" && <BatchInference models={predictorModels} />}
      </section>

      {uploadOpen && (
        <UploadModal onClose={() => setUploadOpen(false)}>
          <UploadModel
            onUploaded={(next) => {
              setModels(next);
              setUploadOpen(false);
            }}
          />
        </UploadModal>
      )}
    </main>
  );
}
