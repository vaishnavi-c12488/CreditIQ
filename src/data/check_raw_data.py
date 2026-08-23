from pathlib import Path


RAW_DIR = Path("data/raw")

csv_files = sorted(RAW_DIR.glob("*.csv"))

print(f"CSV files found: {len(csv_files)}")
print("-" * 70)

for file in csv_files:
    with file.open("r", encoding="utf-8", errors="ignore") as f:
        row_count = sum(1 for _ in f) - 1

    print(f"{file.name:<35} {row_count:>12,} rows")

print("-" * 70)