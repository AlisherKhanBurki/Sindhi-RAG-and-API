import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_engine import SindhiRAGEngine

print("[1] Initializing Sindhi RAG Engine...", flush=True)
engine = SindhiRAGEngine()

# Test 1: In-domain query (Should hit RAG)
q1 = "شاهه لطيف جي سُر ڪلياڻ جو پهريون بيت ۽ ان جي تشريح ڇا آهي؟"
print(f"\n[2] Testing Query 1 (In-domain): {q1}", flush=True)
t0 = time.time()
r1 = engine.query(q1)
print(f"Elapsed: {time.time()-t0:.1f}s | Mode: {r1['mode']} | Score: {r1['confidence_score']:.2f} | Model: {r1['model_used']}", flush=True)
print("Answer:\n" + r1["answer"], flush=True)
if r1.get("retrieved_documents"):
    print("\nTop Cited Document:")
    print("  Source:", r1["retrieved_documents"][0]["source"])
    print("  Melody:", r1["retrieved_documents"][0].get("melody"))
    print("  Similarity Score:", f"{r1['retrieved_documents'][0]['similarity_score']:.2f}")

# Test 2: Out-of-domain query (Should trigger Gemini API Fallback)
q2 = "ڪوانٽم ڪمپيوٽر ڇا آهي ۽ اهو روايتي ڪمپيوٽر کان ڪيئن مختلف آهي؟"
print(f"\n[3] Testing Query 2 (Out-of-domain / Fallback): {q2}", flush=True)
t0 = time.time()
r2 = engine.query(q2)
print(f"Elapsed: {time.time()-t0:.1f}s | Mode: {r2['mode']} | Score: {r2['confidence_score']:.2f} | Model: {r2['model_used']}", flush=True)
if r2.get("fallback_reason"):
    print("Fallback Reason:", r2["fallback_reason"])
print("Answer:\n" + r2["answer"], flush=True)
