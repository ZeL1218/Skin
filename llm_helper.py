import os, json, requests

def have_ollama():
  try:
    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=0.5)
    return r.status_code == 200
  except Exception:
    return False

def ollama_generate(model: str, prompt: str, temperature: float = 0.3, num_ctx: int = 2048, timeout: int = 120) -> str:
  url = "http://127.0.0.1:11434/api/generate"
  payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}}
  r = requests.post(url, json=payload, timeout=timeout)
  r.raise_for_status()
  data = r.json()
  return data.get("response", "").strip()

def have_llama_cpp(model_path: str) -> bool:
  if not os.path.exists(model_path): return False
  try:
    import llama_cpp  # noqa
    return True
  except Exception:
    return False

_llm_cache = None
def _load_llama_cpp(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
  global _llm_cache
  if _llm_cache is not None:
    return _llm_cache
  from llama_cpp import Llama
  _llm_cache = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=0, logits_all=False)
  return _llm_cache

def llama_cpp_generate(model_path: str, prompt: str, temperature: float = 0.3, n_ctx: int = 2048, n_gpu_layers: int = -1, max_tokens: int = 256) -> str:
  llm = _load_llama_cpp(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
  out = llm.create_completion(prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=["</s>"])
  txt = out["choices"][0]["text"]
  return txt.strip()
