import shap


class ShapExplainer:
    """Wraps KernelExplainer; explainer is built at training time and fully serialized."""

    def __init__(self, pipeline, background, feature_names):
        self.pipeline = pipeline
        self.background = background
        self.feature_names = feature_names
        self._explainer = shap.KernelExplainer(self._predict_fn, self.background)

    def _predict_fn(self, X):
        return self.pipeline.named_steps["model"].decision_function(
            self.pipeline.named_steps["scaler"].transform(X)
        )

    def explain(self, X):
        """Returns list of dicts mapping feature name to SHAP value."""
        values = self._explainer.shap_values(X)
        return [dict(zip(self.feature_names, row)) for row in values]
