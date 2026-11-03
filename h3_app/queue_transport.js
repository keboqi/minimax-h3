// Gradio parses the same SSE bytes locally; Cloudflare carries only WebSockets.
(() => {
  if (window.fetch.h3QueueWebSocket) return;
  // The server keeps a dropped stream this long (RESUME_SECONDS in queue_transport.py).
  const RESUME_MS = 55000;
  // Closes that end a stream; any other close is a dropped connection to resume.
  const FINAL_CLOSES = new Set([1000, 1008, 1011, 4404]);
  // A new WebSocket through a remote proxy costs a whole connection setup (seconds on a slow
  // link), and Gradio opens a queue stream for every batch of events. So once a stream ends,
  // the next one's socket opens in the background and waits (SPARE_SECONDS on the server).
  const spares = new Map();
  // A spare that lasted this long timed out idle; one that closed sooner is not retried.
  const SPARE_RETRY_MS = 30000;
  const openSpare = (url) => {
    if (spares.has(url)) return;
    const spare = new WebSocket(url);
    const opened = Date.now();
    spare.binaryType = "arraybuffer";
    spares.set(url, spare);
    spare.addEventListener("close", () => {
      if (spares.get(url) !== spare) return;
      spares.delete(url);
      if (Date.now() - opened > SPARE_RETRY_MS) openSpare(url);
    });
  };
  const takeSpare = (url) => {
    const spare = spares.get(url);
    spares.delete(url);
    return spare && spare.readyState <= WebSocket.OPEN ? spare : null;
  };
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
      // The heartbeat stream stays open for the whole page, so only queue streams keep a spare.
      const spareUrl = url.pathname === "/gradio_api/queue/data" ? socketUrl.href : null;
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
        if (spareUrl) openSpare(spareUrl);
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
        const current = socket = (spareUrl && !received && takeSpare(spareUrl)) || new WebSocket(socketUrl);
        current.binaryType = "arraybuffer";
        const start = () => {
          droppedAt = null;
          retries = 0;
          current.send(JSON.stringify({authorization, stream, offset: received}));
        };
        if (current.readyState === WebSocket.OPEN) start();
        else current.onopen = start;
        current.onmessage = (event) => {
          if (finished || current !== socket) return;
          if (typeof event.data === "string") {
            let metadata;
            try { metadata = JSON.parse(event.data); } catch (error) { fail(error); return; }
            // The server states its close here: a proxy can turn the close frame itself into
            // a dropped connection, which would resume a finished stream and fail it.
            if ("end" in metadata) {
              if (metadata.end === 1000 && responded) {
                finished = true;
                cleanup();
                controller.close();
                current.close(1000);
              } else if (metadata.end !== 4000) {
                fail(new TypeError("Generation progress connection closed unexpectedly"));
              }
              return;
            }
            // A resumed stream repeats its response metadata; the response already exists.
            if (responded) return;
            try {
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
