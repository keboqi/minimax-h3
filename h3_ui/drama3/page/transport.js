export class StaleResponse extends Error {}

export function createApi(base, scope) {
  return async function api(path, options = {}) {
    const {unscoped = false, ...request} = options;
    const before = scope();
    const init = {...request, headers: {"Content-Type": "application/json", ...(request.headers || {})}};
    if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
    const response = await fetch(`${base}${path}`, init);
    let data = {};
    try { data = await response.json(); } catch { /* Empty or failed response. */ }
    if (!unscoped && before !== scope()) throw new StaleResponse("The project changed while the request was running.");
    if (!response.ok || data.error) throw new Error(data.error || data.detail || `Request failed (${response.status}).`);
    return data;
  };
}

export function createPoller(step, {delay = 1500, failed = () => {}, scope = () => null, retry = true} = {}) {
  let timer = null;
  let generation = 0;
  function stop() { generation++; clearTimeout(timer); timer = null; }
  function again() {
    clearTimeout(timer);
    const token = generation, owner = scope();
    timer = setTimeout(async () => {
      if (token !== generation || owner !== scope()) return;
      const next = () => { if (token === generation && owner === scope()) again(); };
      try { await step(next); }
      catch (error) { if (token !== generation || owner !== scope()) return; failed(error); if (retry) next(); }
    }, delay);
  }
  again.stop = stop;
  return again;
}
