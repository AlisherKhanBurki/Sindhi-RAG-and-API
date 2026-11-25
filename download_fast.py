"""Fast download of Sindhi datasets using huggingface_hub snapshot_download.
Downloads raw files directly instead of loading through datasets library."""
import os, json, time
from huggingface_hub import snapshot_download

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(BASE, exist_ok=True)

DATASETS = [
    {
        "repo_id": "SindhiLanguageorg/Sindhi-Open-Lexicon",
        "folder": "sindhi_open_lexicon",
        "desc": "223K+ lexical entries"
    },
    {
        "repo_id": "ambile-official/Shah_Jo_Risalo_labeld",
        "folder": "shah_jo_risalo",
        "desc": "43K+ classical poetry verses"
    },
    {
        "repo_id": "ambile-official/Sindhi_Mega_Corpus_118_Million_Tokens",
        "folder": "sindhi_mega_corpus_118m",
        "desc": "118M token mega corpus"
    },
    {
        "repo_id": "aakashMeghwar01/sindhi-corpus-505m",
        "folder": "sindhi_corpus_505m",
        "desc": "505M token corpus"
    },
]

results = {}
for ds_info in DATASETS:
    repo_id = ds_info["repo_id"]
    local_dir = os.path.join(BASE, ds_info["folder"])
    
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Description: {ds_info['desc']}")
    print(f"Saving to:   {local_dir}")
    print(f"{'='*60}", flush=True)
    
    start = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        elapsed = time.time() - start
        
        # Count files and total size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(path):
            # Skip .cache and hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        
        size_mb = round(total_size / (1024*1024), 2)
        results[repo_id] = {
            "status": "SUCCESS",
            "path": path,
            "files": file_count,
            "size_mb": size_mb,
            "time": f"{elapsed:.1f}s"
        }
        print(f"  ✓ Done! {file_count} files, {size_mb} MB in {elapsed:.1f}s", flush=True)
        
    except Exception as e:
        elapsed = time.time() - start
        results[repo_id] = {"status": "FAILED", "error": str(e), "time": f"{elapsed:.1f}s"}
        print(f"  ✗ FAILED after {elapsed:.1f}s: {e}", flush=True)

# Save summary
summary_path = os.path.join(BASE, "download_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print("DOWNLOAD SUMMARY")
print(f"{'='*60}")
for repo_id, res in results.items():
    if res["status"] == "SUCCESS":
        print(f"  ✓ {repo_id}: {res['files']} files, {res['size_mb']} MB ({res['time']})")
    else:
        print(f"  ✗ {repo_id}: {res['error']} ({res['time']})")
print(f"{'='*60}")
print(f"Summary saved to: {summary_path}")
