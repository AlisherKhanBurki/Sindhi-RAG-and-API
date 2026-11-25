import os
from google import genai
from config import GEMINI_API_KEY, GENERATION_MODELS
from retriever import SindhiRetriever

class SindhiRAGEngine:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file or environment variables.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.retriever = SindhiRetriever()

    def generate_with_fallback(self, prompt: str, system_instruction: str = None):
        """
        Calls Gemini models with automatic fallback across models.
        """
        last_error = None
        for model in GENERATION_MODELS:
            try:
                cfg = {}
                if system_instruction:
                    cfg["system_instruction"] = system_instruction
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=cfg if cfg else None
                )
                if resp.text:
                    return resp.text.strip(), model
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All generation models failed. Last error: {last_error}")

    def query(self, user_question: str) -> dict:
        """
        End-to-End RAG query flow with automatic Gemini API fallback:
        1. Retrieve top-k documents from FAISS index.
        2. If relevance score is low (< threshold), trigger Gemini API fallback directly.
        3. If relevance is sufficient, prompt Gemini to answer strictly from retrieved Sindhi text.
        4. If RAG context fails to answer adequately, trigger Gemini API fallback.
        """
        retrieval = self.retriever.search(user_question)
        top_score = retrieval["top_score"]
        is_confident = retrieval["is_confident"]
        docs = retrieval["documents"]

        # Case 1: Low similarity score -> Trigger Gemini API Fallback
        if not is_confident or len(docs) == 0:
            return self._execute_gemini_fallback(
                user_question,
                reason=f"گھٽ مطابقت جو اسڪور ({top_score:.2f} < {retrieval['threshold']})",
                retrieval_data=retrieval
            )

        # Case 2: Sufficient similarity -> Attempt Grounded RAG Generation
        context_blocks = []
        for i, doc in enumerate(docs):
            context_blocks.append(
                f"[دستاويز {i+1} | ذريعو: {doc['source']} | سر: {doc.get('melody', 'نامعلوم')} | اسڪور: {doc['similarity_score']:.2f}]\n"
                f"{doc['content']}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        rag_prompt = f"""توهان هڪ سنڌي ادبي ماهر آهيو. هيٺ ڏنل سنڌي ماخذن (دستاويزن) جي روشنيءَ ۾ سوال جو صحيح ۽ جامع جواب سنڌي ٻوليءَ ۾ ڏيو.
جيڪڏهن ڏنل دستاويزن مان سوال جو صحيح جواب واضح طور تي نه ٿو ملي، ته صرف ۽ صرف لکو: [INSUFFICIENT_CONTEXT]

ڏنل سنڌي دستاويز:
{context_str}

استعمال ڪندڙ جو سوال:
{user_question}

جواب سنڌي ٻوليءَ ۾:"""

        system_instruction = "توهان هڪ مستند سنڌي ريسرچ اسسٽنٽ آهيو. صرف مهيا ڪيل ماخذن تي ٻڌل جواب ڏيو."

        try:
            answer, model_used = self.generate_with_fallback(rag_prompt, system_instruction)
            
            # Check if RAG context was deemed insufficient by the model
            clean_ans = answer.strip()
            if clean_ans.startswith("[INSUFFICIENT_CONTEXT]") or len(clean_ans) < 20:
                return self._execute_gemini_fallback(
                    user_question,
                    reason="RAG دستاويز مان سوال جو مڪمل حل نه ملي سگهيو",
                    retrieval_data=retrieval
                )

            return {
                "question": user_question,
                "answer": answer,
                "mode": "RAG_GROUNDED",
                "confidence_score": top_score,
                "model_used": model_used,
                "retrieved_documents": docs
            }

        except Exception as e:
            # If RAG generation encounters an error, fallback to broad Gemini generation
            return self._execute_gemini_fallback(
                user_question,
                reason=f"RAG جواب دوران خرابي: {str(e)}",
                retrieval_data=retrieval
            )

    def _execute_gemini_fallback(self, user_question: str, reason: str, retrieval_data: dict) -> dict:
        """
        Direct Gemini API Fallback:
        Called when local RAG context is missing, weak, or deemed not good.
        """
        fallback_prompt = f"""استعمال ڪندڙ جو سوال آهي:
"{user_question}"

مهرباني ڪري هڪ قابل ۽ مستند سنڌي ماهر ۽ محقق طور هن سوال جو تفصيلي، جامع ۽ اثرائتو جواب باوقار سنڌي ٻوليءَ ۾ ڏيو."""

        system_instruction = "توهان سنڌ، سنڌي ٻولي، تاريخ ۽ ادب جا هڪ نامور محقق ۽ ماهر آهيو. هميشه صحيح، مٺي ۽ معياري سنڌي ٻوليءَ ۾ جواب ڏيو."

        answer, model_used = self.generate_with_fallback(fallback_prompt, system_instruction)

        disclaimer = (
            f"⚡ [نوٽ: هي جواب RAG ڊيٽابيس مان مڪمل نه مليو ({reason})، "
            f"تنهنڪري هي جواب سڌو سنئون Gemini API جي وسيع علمي ماخذ مان تيار ڪيو ويو آهي:]\n\n"
        )

        return {
            "question": user_question,
            "answer": disclaimer + answer,
            "mode": "GEMINI_FALLBACK",
            "fallback_reason": reason,
            "confidence_score": retrieval_data.get("top_score", 0.0),
            "model_used": model_used,
            "retrieved_documents": retrieval_data.get("documents", [])
        }
