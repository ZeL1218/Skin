import os, sys, subprocess, time, threading, socket
import webview

def here():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE = here()
os.chdir(BASE)

VENV_PY = os.path.join(BASE, ".venv", "Scripts", "python.exe")
PYTHON = VENV_PY if os.path.exists(VENV_PY) else sys.executable
URL = "http://127.0.0.1:8501"
CMD = [PYTHON, "-m", "streamlit", "run", "app.py",
       "--server.port=8501", "--server.address=127.0.0.1",
       "--server.headless=true"]

ENV = os.environ.copy()
ENV["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

proc = None

def wait_port(host, port, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            time.sleep(0.2)
    return False

def start_streamlit():
    global proc
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    proc = subprocess.Popen(CMD, env=ENV, cwd=BASE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=creationflags)
    wait_port("127.0.0.1", 8501)

def on_closed():
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
    except Exception:
        pass

def bootstrap(window):
    th = threading.Thread(target=start_streamlit, daemon=True)
    th.start()
    th.join()
    try:
        window.load_url(URL)
    except Exception:
        pass

if __name__ == "__main__":
    w = webview.create_window("Skin Lesion AI", html="<h3 style='font-family:system-ui;margin:2rem'>Loading...</h3>",
                              width=1100, height=780, min_size=(900,600), confirm_close=False)
    w.events.closing += on_closed
    try:
        webview.start(func=bootstrap, args=(w,), gui="edgechromium", debug=False)
    except Exception:
        webview.start(func=bootstrap, args=(w,), debug=False)
    on_closed()