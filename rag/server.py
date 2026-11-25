import os
import sys
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from rag_engine import SindhiRAGEngine

PORT = int(os.environ.get("PORT", 7860))
WEB_DIR = os.path.join(BASE_DIR, "web")

print("Initializing Sindhi RAG Engine for Web Server...")
engine = SindhiRAGEngine()
print("RAG Engine ready.")

class RAGHttpHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            html_path = os.path.join(WEB_DIR, "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, "text/html; charset=utf-8")
                self.wfile.write(content)
            except Exception as e:
                self._set_headers(500, "text/plain")
                self.wfile.write(str(e).encode("utf-8"))
        elif self.path == "/api/status":
            status_data = {
                "status": "online",
                "indexed_documents": engine.retriever.index.ntotal,
                "embedding_model": "bge-m3 (NVIDIA RTX A4000 GPU)",
                "default_generation_model": "gemini-3.5-flash"
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self._set_headers(404, "text/plain")
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                req_json = json.loads(body)
                query = req_json.get("query", "").strip()

                if not query:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"status": "error", "error": "Query cannot be empty"}).encode("utf-8"))
                    return

                print(f"Incoming query: {query}", flush=True)
                result = engine.query(query)

                self._set_headers(200)
                response_json = json.dumps({"status": "success", "data": result}, ensure_ascii=False)
                self.wfile.write(response_json.encode("utf-8"))

            except Exception as e:
                print(f"Error handling /api/chat: {e}", flush=True)
                self._set_headers(500)
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"status": "error", "error": "Endpoint not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Override to keep logs clean
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = ThreadingHTTPServer(server_address, RAGHttpHandler)
    print(f"\n=======================================================")
    print(f" Sindhi RAG Web Server is running!")
    print(f" URL: http://localhost:{PORT}")
    print(f"=======================================================\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
