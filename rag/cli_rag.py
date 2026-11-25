import sys
from rag_engine import SindhiRAGEngine

def main():
    print("=" * 65)
    print("      سنڌي ادبي ۽ تحقيقي RAG سسٽم (Sindhi RAG Engine)")
    print("   Powered by Local FAISS Index + Gemini API Smart Fallback")
    print("=" * 65)
    print("Loading RAG Engine and FAISS Index...")
    engine = SindhiRAGEngine()
    print("System is Ready! Type your query in Sindhi or English.")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            query = input("\n📝 سوال لکو (Enter your question): ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "بند"]:
                print("الله واهي! بند ڪيو ويو.")
                break

            print("\n🔍 تلاش ڪندي ۽ جواب تيار ڪندي...", flush=True)
            res = engine.query(query)

            print("\n" + "=" * 65)
            print(f"📊 ڪم جو طريقو (Mode): {res['mode']}")
            print(f"📈 اعتمادي اسڪور (Confidence Score): {res['confidence_score']:.2f}")
            print(f"🤖 استعمال ٿيل ماڊل (Model Used): {res['model_used']}")
            print("=" * 65)
            print("\n💡 جواب (Answer):")
            print(res["answer"])

            if res["mode"] == "RAG_GROUNDED" and res.get("retrieved_documents"):
                print("\n📚 ماخذ (Top Cited Sources):")
                for i, doc in enumerate(res["retrieved_documents"][:2]):
                    print(f"  {i+1}. {doc['source']} - {doc.get('melody', '')} (اسڪور: {doc['similarity_score']:.2f})")
            print("=" * 65)

        except KeyboardInterrupt:
            print("\nخدا حافظ!")
            break
        except Exception as e:
            print(f"\n❌ خرابي: {e}")

if __name__ == "__main__":
    main()
