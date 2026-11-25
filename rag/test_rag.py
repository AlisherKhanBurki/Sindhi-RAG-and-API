import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import SindhiRAGEngine

def run_tests():
    print("=" * 70)
    print("   TESTING SINDHI RAG SYSTEM + GEMINI API SMART FALLBACK")
    print("=" * 70)

    engine = SindhiRAGEngine()

    test_queries = [
        {
            "title": "TEST 1: In-domain Query (Shah Jo Risalo - Sur Kalyan)",
            "query": "شاهه لطيف جي سُر ڪلياڻ جي پهرئين بيت جي تشريح ڇا آهي؟",
            "expected_mode": "RAG_GROUNDED"
        },
        {
            "title": "TEST 2: Out-of-domain Query (Modern Tech / Quantum Computing)",
            "query": "ڪوانٽم ڪمپيوٽر ڇا آهي ۽ اهو روايتي ڪمپيوٽر کان ڪيئن مختلف آهي؟",
            "expected_mode": "GEMINI_FALLBACK"
        }
    ]

    for test in test_queries:
        print("\n" + "#" * 70)
        print(f"👉 {test['title']}")
        print(f"📝 سوال (Query): {test['query']}")
        print("#" * 70)

        result = engine.query(test["query"])

        print(f"\n📊 طريقو (Mode): {result['mode']}")
        print(f"🎯 اعتمادي اسڪور (Confidence Score): {result['confidence_score']:.2f}")
        print(f"🤖 ماڊل (Model): {result['model_used']}")
        if result['mode'] == 'GEMINI_FALLBACK':
            print(f"⚠️ ڇو Gemini ڏانهن منتقل ٿيو (Fallback Reason): {result.get('fallback_reason')}")

        print("\n💬 جواب (Generated Answer):\n")
        print(result["answer"])

        if result.get("retrieved_documents"):
            print("\n📚 ماخذ (Top Retrieved Docs):")
            for doc in result["retrieved_documents"][:2]:
                print(f"  • {doc['source']} | {doc.get('melody', '')} | اسڪور: {doc['similarity_score']:.2f}")

        print("\n" + "-" * 70)

if __name__ == "__main__":
    run_tests()
