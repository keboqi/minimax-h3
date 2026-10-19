// Gradio parses the same SSE bytes locally; Cloudflare carries only WebSockets.
(() => {
  if (window.fetch.h3QueueWebSocket) return;
  // The server keeps a dropped stream this long (RESUME_SECONDS in queue_transport.py).
  const RESUME_MS = 55000;
  // Closes that end a stream; any other close is a dropped connection to resume.
  const FINAL_CLOSES = new Set([1000, 1008, 1011, 4404]);
  const originalFetch = window.fetch.bind(window);
  const streamId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)),
    (byte) => byte.toString(16).padStart(2, "0")).join("");
  const bridgeFetch = (input, init) => {
    const url = new URL(input instanceof Request ? input.url : input, location.href);
    const matches = url.pathname === "/gradio_api/queue/data" ||
      /^\/gradio_api\/heartbeat\/[^/]+$/.test(url.pathname);
    if (url.origin !== location.origin || !matches) return originalFetch(input, init);
    const request = new Request(input, init);
    // WebSockets attach browser cookies, so preserve explicit anonymous fetches.
    if (request.method !== "GET" || request.credentials === "omit") return originalFetch(input, init);
    return new Promise((resolve, reject) => {
      if (request.signal.aborted) {
        reject(request.signal.reason);
        return;
      }
      const socketUrl = new URL(url);
      socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const stream = streamId();
      const authorization = request.headers.get("authorization");
      let socket;
      let controller;
      let responded = false;
      let finished = false;
      let received = 0;
      let droppedAt = null;
      let retries = 0;
      let retryTimer = null;
      const cleanup = () => {
        clearTimeout(retryTimer);
        request.signal.removeEventListener("abort", abort);
      };
      const fail = (error) => {
        if (finished) return;
        finished = true;
        cleanup();
        if (responded) controller.error(error);
        else reject(error);
        socket.close(1000);
      };
      const abort = () => fail(request.signal.reason || new DOMException("Aborted", "AbortError"));
      const body = new ReadableStream({
        start(value) { controller = value; },
        cancel() {
          finished = true;
          cleanup();
          socket.close(1000);
        },
      });
      const connect = () => {
        const current = socket = new WebSocket(socketUrl);
        current.binaryType = "arraybuffer";
        current.onopen = () => {
          droppedAt = null;
          retries = 0;
          current.send(JSON.stringify({authorization, stream, offset: received}));
        };
        current.onmessage = (event) => {
          if (finished || current !== socket) return;
          if (typeof event.data === "string") {
            // A resumed stream repeats its response metadata; the response already exists.
            if (responded) return;
            try {
              const metadata = JSON.parse(event.data);
              const response = new Response(body, {
                status: metadata.status,
                headers: {"Content-Type": metadata.content_type},
              });
              responded = true;
              resolve(response);
            } catch (error) { fail(error); }
          } else {
            controller.enqueue(new Uint8Array(event.data));
            received += event.data.byteLength;
            current.send(JSON.stringify({ack: received}));
          }
        };
        current.onclose = (event) => {
          if (finished || current !== socket) return;
          if (event.code === 1000 && responded) {
            finished = true;
            cleanup();
            controller.close();
            return;
          }
          droppedAt ??= Date.now();
          if (FINAL_CLOSES.has(event.code) || Date.now() - droppedAt > RESUME_MS) {
            fail(new TypeError("Generation progress connection closed unexpectedly"));
            return;
          }
          // Gradio reloads the page when this stream fails, so reconnect quietly instead.
          retryTimer = setTimeout(connect, Math.min(500 * 2 ** retries++, 5000));
        };
      };
      request.signal.addEventListener("abort", abort, {once: true});
      connect();
    });
  };
  bridgeFetch.h3QueueWebSocket = true;
  window.fetch = bridgeFetch;
})();
