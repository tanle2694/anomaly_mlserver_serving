"""
Generate a sample CSV with normal sensor readings and a small anomaly fraction.

Usage:
    python generate_sample_data.py                        # default output
    python generate_sample_data.py --out data/sensors.csv
    python generate_sample_data.py --samples 5000 --anomalies 100 --seed 7
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sample_data.csv",
                        help="Output CSV path")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of normal rows")
    parser.add_argument("--anomalies", type=int, default=50,
                        help="Number of anomaly rows")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start", default="2024-01-01T00:00:00",
                        help="Timestamp of the first row (ISO 8601)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between readings (default: 60)")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    total = args.samples + args.anomalies

    normal = pd.DataFrame({
        "temperature": rng.normal(70, 5, args.samples),
        "pressure":    rng.normal(30, 2, args.samples),
        "vibration":   rng.normal(0.5, 0.1, args.samples),
        "humidity":    rng.normal(40, 10, args.samples),
    })

    anomalies = pd.DataFrame({
        "temperature": rng.normal(100, 10, args.anomalies),
        "pressure":    rng.normal(50, 5, args.anomalies),
        "vibration":   rng.normal(1.5, 0.3, args.anomalies),
        "humidity":    rng.normal(10, 5, args.anomalies),
    })

    df = pd.concat([normal, anomalies], ignore_index=True).sample(
        frac=1, random_state=args.seed
    ).reset_index(drop=True)

    start = datetime.fromisoformat(args.start)
    df.insert(0, "timestamp", [
        (start + timedelta(seconds=i * args.interval)).isoformat()
        for i in range(total)
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(
        f"Written {total} rows "
        f"({args.samples} normal + {args.anomalies} anomalies) → {out}"
    )
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
