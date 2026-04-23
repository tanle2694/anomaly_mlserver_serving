import argparse

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    RANDOM_SEED,
    build_metadata,
    build_shap_explainer,
    evaluate,
    generate_sample_data,
    load_data,
    save,
    test,
    train,
)


# =========================
# 1. BUILD PIPELINE
# =========================
def build_pipeline(contamination="auto"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=RANDOM_SEED,
        )),
    ])


# =========================
# 2. MAIN
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="iforest_v1")
    parser.add_argument("--data", help="Path to training CSV file; omit to generate synthetic data", default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["Chiller Power (kW)","Chilled Water(CHW) Supply Temp (C°)","Chilled Water(CHW) Return Temp (C°)","Condensed Water(CDW) Supply Temp (C°)","Condensed Water(CDW) Return Temp (C°)","Chilled Water(CDW) Return Flow (m3/hr)","Condensed Water(CDW) Return Flow (m3/hr)","Cooling Tower Power (kW)","Condensed Water(CDW) Pump Power (kW)","Chilled Water(CHW) Pump Power (kW)","Cooling Load (RT)"],
        help="Feature column names to use from the CSV (default: temperature pressure vibration humidity)",
    )
    parser.add_argument("--contamination", default="auto",
                        help="IsolationForest contamination: 'auto' or float 0–0.5 (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()

    contamination = args.contamination
    if contamination != "auto":
        contamination = float(contamination)

    features = args.features

    print("Loading data...")
    if args.data:
        df = load_data(args.data, features)
    else:
        df = generate_sample_data(features)

    print("Building Isolation Forest pipeline...")
    pipeline = build_pipeline(contamination)

    print("Training...")
    detector = train(pipeline, df, features)

    print("Evaluating...")
    df = evaluate(detector, df, features)

    print("Building SHAP...")
    explainer = build_shap_explainer(detector.pipeline, df, features)

    print("Saving...")
    metadata = build_metadata(
        model_name=args.model_name,
        features=features,
        df=df,
        data_source=args.data or "synthetic",
        pipeline=detector.pipeline,
        anomaly_rule={"operator": "<", "threshold": 0.0},
    )
    save(detector, explainer, args.model_name, features, metadata)

    print("Testing...")
    test(detector, explainer, df, features)


if __name__ == "__main__":
    main()
