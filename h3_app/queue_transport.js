// Gradio parses the same SSE bytes locally; Cloudflare carries only WebSockets.
(() => {
  if (window.fetch.h3QueueWebSocket) return;
  const originalFetch = window.fetch.bind(window);
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
      const socket = new WebSocket(socketUrl);
      socket.binaryType = "arraybuffer";
      let controller;
      let responded = false;
      let finished = false;
      const cleanup = () => request.signal.removeEventListener("abort", abort);
      const fail = (error) => {
        if (finished) return;
        finished = true;
        cleanup();
        if (responded) controller.error(error);
        else reject(error);
        socket.close();
      };
      const abort = () => fail(request.signal.reason || new DOMException("Aborted", "AbortError"));
      const body = new ReadableStream({
        start(value) { controller = value; },
        cancel() {
          finished = true;
          cleanup();
          socket.close();
        },
      });
      request.signal.addEventListener("abort", abort, {once: true});
      socket.onopen = () => socket.send(JSON.stringify({
        authorization: request.headers.get("authorization"),
      }));
      socket.onmessage = (event) => {
        if (finished) return;
        if (typeof event.data === "string") {
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
        }
      };
      socket.onerror = () => fail(new TypeError("Generation progress connection failed"));
      socket.onclose = (event) => {
        if (finished) return;
        if (!responded || event.code !== 1000) {
          fail(new TypeError("Generation progress connection closed unexpectedly"));
          return;
        }
        finished = true;
        cleanup();
        controller.close();
      };
    });
  };
  bridgeFetch.h3QueueWebSocket = true;
  window.fetch = bridgeFetch;
})();
