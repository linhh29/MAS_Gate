"""Single-process local Skywork reward server for the opt-in ablation."""
from __future__ import annotations
import json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.environ.get("MAS_GATE_SKYWORK_MODEL_PATH", "./models/Skywork-Reward-V2-Qwen3-8B")
PORT = int(os.environ.get("MAS_GATE_SKYWORK_REWARD_PORT", "9480"))
MAX_CONCURRENCY = max(1, int(os.environ.get("MAS_GATE_SKYWORK_MAX_CONCURRENCY", "2")))
_SCORE_GATE = threading.BoundedSemaphore(MAX_CONCURRENCY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", local_files_only=True)
model.eval()

def score(question, answer):
    # The model is a single CPU-backed classifier. Bound inference concurrency;
    # an unbounded ThreadingHTTPServer otherwise creates hundreds of competing
    # PyTorch calls and makes callers hit their HTTP timeout.
    with _SCORE_GATE:
        return _score_impl(question, answer)

def _score_impl(question, answer):
    conv = [{"role":"user", "content":str(question)}, {"role":"assistant", "content":str(answer)}]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token): text = text[len(tokenizer.bos_token):]
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=16384)
    with torch.inference_mode(): return float(model(**batch).logits[0][0].item())

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health": self._send({"status":"ok", "model":MODEL_PATH})
        else: self.send_error(404)
    def do_POST(self):
        if self.path != "/score": self.send_error(404); return
        try:
            n=int(self.headers.get("Content-Length", "0")); obj=json.loads(self.rfile.read(n)); self._send({"score":score(obj["question"], obj["answer"])})
        except Exception as exc: self._send({"error":f"{type(exc).__name__}: {exc}"}, 500)
    def _send(self,obj,status=200):
        data=json.dumps(obj).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self,*args): pass

print(f"[skywork_reward_server] model={MODEL_PATH} port={PORT} max_concurrency={MAX_CONCURRENCY}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
