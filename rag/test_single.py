import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_engine import SindhiRAGEngine

engine = SindhiRAGEngine()
res = engine.query("اوَلِ اَللهُ عَلِيمُ بيت جي تشريح ڇا آهي؟")
print("=" * 60)
print("Mode:", res["mode"])
print(f"Confidence Score: {res['confidence_score']:.2f}")
print("Model Used:", res["model_used"])
print("=" * 60)
print("Answer:\n" + res["answer"])
if res.get("retrieved_documents"):
    print("\nTop Source:", res["retrieved_documents"][0]["source"], "| Melody:", res["retrieved_documents"][0].get("melody"))
