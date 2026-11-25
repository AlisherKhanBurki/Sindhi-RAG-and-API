"""Download Sindhi datasets from Hugging Face for RAG pipeline."""
import os, sys, json, time

# Ensure output directory exists
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(BASE, exist_ok=True)

from datasets import load_dataset

DATASETS = [
    {
        "id": "SindhiLanguageorg/Sindhi-Open-Lexicon",
        "folder": "sindhi_open_lexicon",
        "desc": "223K+ lexical entries"
    },
    {
        "id": "ambile-official/Shah_Jo_Risalo_labeld",
        "folder": "shah_jo_risalo",
        "desc": "43K+ classical poetry verses"
    },
    {
        "id": "ambile-official/Sindhi_Mega_Corpus_118_Million_Tokens",
        "folder": "sindhi_mega_corpus_118m",
        "desc": "118M token mega corpus"
    },
    {
        "id": "aakashMeghwar01/sindhi-corpus-505m",
        "folder": "sindhi_corpus_505m",
        "desc": "505M token corpus"
    },
]

results = {}
for ds_info in DATASETS:
    ds_id = ds_info["id"]
    folder = os.path.join(BASE, ds_info["folder"])
    os.makedirs(folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {ds_id}")
    print(f"Description: {ds_info['desc']}")
    print(f"Saving to:   {folder}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        ds = load_dataset(ds_id)
        elapsed = time.time() - start
        print(f"Downloaded in {elapsed:.1f}s")
        
        # Save each split
        for split_name, split_data in ds.items():
            out_path = os.path.join(folder, f"{split_name}.jsonl")
            print(f"  Saving {split_name} ({len(split_data)} rows) -> {out_path}")
            split_data.to_json(out_path, force_ascii=False)
        
        # Save metadata
        meta = {
            "dataset_id": ds_id,
            "description": ds_info["desc"],
            "splits": {name: len(data) for name, data in ds.items()},
            "columns": {name: list(data.column_names) for name, data in ds.items()},
            "download_time_seconds": round(elapsed, 1)
        }
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        results[ds_id] = {"status": "SUCCESS", "rows": meta["splits"], "time": f"{elapsed:.1f}s"}
        print(f"  ✓ Done!")
        
    except Exception as e:
        elapsed = time.time() - start
        results[ds_id] = {"status": "FAILED", "error": str(e), "time": f"{elapsed:.1f}s"}
        print(f"  ✗ FAILED: {e}")

print(f"\n\n{'='*60}")
print("DOWNLOAD SUMMARY")
print(f"{'='*60}")
for ds_id, res in results.items():
    status = res["status"]
    if status == "SUCCESS":
        total_rows = sum(res["rows"].values())
        print(f"  ✓ {ds_id}: {total_rows:,} rows ({res['time']})")
    else:
        print(f"  ✗ {ds_id}: {res['error']} ({res['time']})")
print(f"{'='*60}")
