import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "127.0.0.1"
PORT = 7860


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agricultural QA Chatbot</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #1f2937;
      --text: #f3f7fb;
      --muted: #aab6c5;
      --line: #344054;
      --accent: #39d98a;
      --accent-2: #5cc8ff;
      --danger: #ff6b6b;
      --user: #164433;
      --user-line: #2f9d70;
      --bot-line: #3d4d63;
      --shadow: rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(57, 217, 138, 0.12), transparent 34%),
        linear-gradient(135deg, #0d1117 0%, #101722 48%, #0a1018 100%);
      color: var(--text);
    }

    .app {
      width: min(1180px, calc(100vw - 48px));
      height: 100vh;
      margin: 0 auto;
      padding: 24px 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 16px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      border: 1px solid rgba(92, 200, 255, 0.22);
      border-radius: 8px;
      padding: 16px 18px;
      background: rgba(21, 27, 35, 0.86);
      box-shadow: 0 10px 32px var(--shadow);
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }

    .title-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .mark {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #123624;
      border: 1px solid #2f9d70;
      color: var(--accent);
      font-weight: 900;
    }

    .status {
      min-width: 150px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--accent);
      text-align: center;
      font-weight: 700;
    }

    main { min-height: 0; }

    .chat-shell {
      height: 100%;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 14px 40px var(--shadow);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto auto;
      overflow: hidden;
    }

    .samples {
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #101722;
    }

    .sample {
      flex: 0 0 auto;
      max-width: 285px;
      border: 1px solid #436170;
      border-radius: 8px;
      padding: 10px 12px;
      background: #122233;
      color: #d7f2ff;
      font-size: 14px;
      line-height: 1.35;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }

    .sample:hover {
      background: #16304a;
      border-color: var(--accent-2);
      transform: translateY(-1px);
    }

    .messages {
      min-height: 0;
      overflow-y: auto;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: #0b1220;
    }

    .message {
      display: grid;
      gap: 6px;
      max-width: 78%;
    }

    .message.user { align-self: flex-end; }
    .message.bot { align-self: flex-start; }

    .sender {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .message.user .sender { text-align: right; }

    .bubble {
      border: 1px solid var(--bot-line);
      border-radius: 8px;
      padding: 13px 15px;
      color: var(--text);
      font-size: 16px;
      line-height: 1.55;
      white-space: pre-wrap;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.22);
    }

    .user .bubble {
      background: var(--user);
      border-color: var(--user-line);
    }

    .bot .bubble { background: #161f2c; }

    .composer {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      padding: 16px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    textarea {
      width: 100%;
      min-height: 54px;
      max-height: 130px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      background: #0b1220;
      color: var(--text);
      font: 16px/1.45 Arial, Helvetica, sans-serif;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent-2);
      box-shadow: 0 0 0 3px rgba(92, 200, 255, 0.18);
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 12px 18px;
      background: var(--accent);
      color: #04130b;
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 120ms ease, filter 120ms ease;
    }

    button:hover {
      filter: brightness(1.05);
      transform: translateY(-1px);
    }

    button.secondary {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      padding: 0 16px 16px;
      background: var(--panel);
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--panel-2);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }

    .metric strong {
      font-size: 16px;
      color: var(--text);
    }

    .error { color: var(--danger); }

    @media (max-width: 820px) {
      .app {
        width: min(100vw - 24px, 720px);
        padding: 14px 0;
      }

      header {
        align-items: start;
        flex-direction: column;
      }

      .status { width: 100%; }
      .composer { grid-template-columns: 1fr; }
      .meta { grid-template-columns: 1fr; }
      .message { max-width: 94%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <div class="title-row">
          <div class="mark">Ag</div>
          <h1>Agricultural QA Chatbot</h1>
        </div>
        <p class="subtitle">Ask crop and farming questions, then review the generated answer in a chat view.</p>
      </div>
      <div class="status" id="status">Ready</div>
    </header>

    <main>
      <section class="chat-shell">
        <div class="samples" id="samples"></div>
        <div class="messages" id="messages">
          <div class="message bot">
            <div class="sender">Assistant</div>
            <div class="bubble">Hi. Ask an agriculture question, or choose one of the sample queries above.</div>
          </div>
        </div>
        <div class="composer">
          <textarea id="question" placeholder="Type your question here..."></textarea>
          <button id="ask">Send</button>
          <button class="secondary" id="clear">Clear</button>
        </div>
        <div class="meta">
          <div class="metric"><span>Question Words</span><strong id="qWords">0</strong></div>
          <div class="metric"><span>Answer Words</span><strong id="aWords">0</strong></div>
          <div class="metric"><span>Response Time</span><strong id="time">-</strong></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const sampleQueries = [
      "How to control stem borer in paddy?",
      "How to treat black gram seeds?",
      "Best fertilizer for wheat",
      "How to control leaf curl in chilli",
      "What fertilizer is suitable for rice crop?",
      "How can I control leaf spot disease in groundnut?",
      "What are common symptoms of nitrogen deficiency in maize?",
      "How often should tomato plants be irrigated?",
      "What is the best way to manage pests in cotton?"
    ];

    const question = document.getElementById("question");
    const messages = document.getElementById("messages");
    const samples = document.getElementById("samples");
    const ask = document.getElementById("ask");
    const clear = document.getElementById("clear");
    const statusEl = document.getElementById("status");
    const qWords = document.getElementById("qWords");
    const aWords = document.getElementById("aWords");
    const time = document.getElementById("time");

    function wordCount(text) {
      return text.trim() ? text.trim().split(/\\s+/).length : 0;
    }

    function addMessage(sender, text, type) {
      const wrapper = document.createElement("div");
      wrapper.className = "message " + type;

      const senderEl = document.createElement("div");
      senderEl.className = "sender";
      senderEl.textContent = sender;

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;

      wrapper.appendChild(senderEl);
      wrapper.appendChild(bubble);
      messages.appendChild(wrapper);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    }

    function setBusy(isBusy) {
      ask.disabled = isBusy;
      statusEl.textContent = isBusy ? "Thinking" : "Ready";
    }

    async function checkAnswer() {
      const text = question.value.trim();
      qWords.textContent = wordCount(text);
      if (!text) {
        addMessage("Assistant", "Please enter a question first.", "bot");
        return;
      }

      setBusy(true);
      addMessage("You", text, "user");
      question.value = "";
      const thinkingBubble = addMessage("Assistant", "Generating answer...", "bot");
      const started = performance.now();

      try {
        const response = await fetch("/api/answer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text })
        });
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Could not generate an answer.");
        }

        thinkingBubble.textContent = data.answer || "No answer returned.";
        aWords.textContent = wordCount(data.answer || "");
      } catch (error) {
        thinkingBubble.textContent = error.message;
        thinkingBubble.classList.add("error");
        aWords.textContent = "0";
      } finally {
        time.textContent = ((performance.now() - started) / 1000).toFixed(2) + "s";
        setBusy(false);
      }
    }

    ask.addEventListener("click", checkAnswer);
    question.addEventListener("input", () => {
      qWords.textContent = wordCount(question.value);
    });
    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        checkAnswer();
      }
    });
    clear.addEventListener("click", () => {
      question.value = "";
      messages.innerHTML = "";
      addMessage("Assistant", "Chat cleared. Ask a new agriculture question or choose a sample query.", "bot");
      qWords.textContent = "0";
      aWords.textContent = "0";
      time.textContent = "-";
      question.focus();
    });

    sampleQueries.forEach((sample) => {
      const button = document.createElement("button");
      button.className = "sample";
      button.type = "button";
      button.textContent = sample;
      button.addEventListener("click", () => {
        question.value = sample;
        qWords.textContent = wordCount(sample);
        question.focus();
      });
      samples.appendChild(button);
    });
  </script>
</body>
</html>
"""


class QAHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/answer":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        question = str(payload.get("question", "")).strip()

        if not question:
            self._send_json(400, {"error": "Question is required."})
            return

        try:
            from inference import generate_answer

            answer = generate_answer(question)
            self._send_json(200, {"answer": answer})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), QAHandler)
    print(f"QA UI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
