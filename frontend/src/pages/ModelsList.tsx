import { useState } from "react";
import { LayoutGrid, List, Trash2, UploadCloud } from "lucide-react";
import { LogicalModel, deleteModel } from "../api/client";

type ViewMode = "card" | "table";

type Props = {
  models: LogicalModel[];
  onModelsChanged: (models: LogicalModel[]) => void;
  onUpload: () => void;
};

export default function ModelsList({ models, onModelsChanged, onUpload }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [confirmTarget, setConfirmTarget] = useState<string | null>(null);

  async function confirmDelete() {
    if (!confirmTarget) return;
    onModelsChanged(await deleteModel(confirmTarget));
    setConfirmTarget(null);
  }

  const loadedCount = models.filter((m) => m.predictor_loaded && m.explainer_loaded).length;

  return (
    <div>
      <div className="models-hero">
        <div className="models-hero-top">
          <span className="models-title">MODELS</span>
          <div className="models-hero-actions">
            <button className="primary-button" type="button" onClick={onUpload}>
              <UploadCloud size={14} />
              Upload model
            </button>
            <div className="view-toggle">
              <button
                className={`view-toggle-btn${viewMode === "card" ? " active" : ""}`}
                type="button"
                onClick={() => setViewMode("card")}
                title="Card view"
              >
                <LayoutGrid size={15} />
              </button>
              <button
                className={`view-toggle-btn${viewMode === "table" ? " active" : ""}`}
                type="button"
                onClick={() => setViewMode("table")}
                title="Table view"
              >
                <List size={15} />
              </button>
            </div>
          </div>
        </div>
        <span className="models-meta">
          {models.length} registered · {loadedCount} loaded
        </span>
      </div>

      {models.length === 0 ? (
        <div className="empty">
          <span className="empty-title">NO MODELS</span>
          upload a model to get started
        </div>
      ) : viewMode === "card" ? (
        <div className="models-grid">
          {models.map((model) => {
            const isLoaded = model.predictor_loaded && model.explainer_loaded;
            return (
              <div key={model.name} className="model-card">
                <div className="model-card-header">
                  <span className="model-card-name">{model.name}</span>
                  <div className={`model-card-status ${isLoaded ? "loaded" : "pending"}`}>
                    <span className={`status-dot ${isLoaded ? "loaded" : "pending"}`} />
                    {isLoaded ? "LOADED" : "PENDING"}
                  </div>
                </div>

                <div className="model-card-artifacts">
                  <span className={model.has_predictor ? "artifact-tag ok" : "artifact-tag"}>
                    predictor
                  </span>
                  <span className={model.has_explainer ? "artifact-tag ok" : "artifact-tag"}>
                    explainer
                  </span>
                </div>

                <div className="model-card-footer">
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => setConfirmTarget(model.name)}
                    title="Delete model"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="models-table-wrap">
          <table className="models-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Predictor</th>
                <th>Explainer</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => {
                const isLoaded = model.predictor_loaded && model.explainer_loaded;
                return (
                  <tr key={model.name}>
                    <td className="model-name">{model.name}</td>
                    <td>
                      <div className={`model-card-status ${isLoaded ? "loaded" : "pending"}`}>
                        <span className={`status-dot ${isLoaded ? "loaded" : "pending"}`} />
                        {isLoaded ? "LOADED" : "PENDING"}
                      </div>
                    </td>
                    <td>
                      <span className={model.has_predictor ? "artifact-tag ok" : "artifact-tag"}>
                        predictor
                      </span>
                    </td>
                    <td>
                      <span className={model.has_explainer ? "artifact-tag ok" : "artifact-tag"}>
                        explainer
                      </span>
                    </td>
                    <td className="actions">
                      <button
                        className="icon-button"
                        type="button"
                        onClick={() => setConfirmTarget(model.name)}
                        title="Delete model"
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {confirmTarget && (
        <div className="modal-overlay" onClick={() => setConfirmTarget(null)}>
          <div className="modal-panel confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="confirm-icon">
              <Trash2 size={22} />
            </div>
            <span className="confirm-title">Delete model?</span>
            <p className="confirm-body">
              <code>{confirmTarget}</code> and all its artifacts will be permanently removed.
            </p>
            <div className="confirm-actions">
              <button className="secondary-button" type="button" onClick={() => setConfirmTarget(null)}>
                Cancel
              </button>
              <button className="danger-button" type="button" onClick={() => void confirmDelete()}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
