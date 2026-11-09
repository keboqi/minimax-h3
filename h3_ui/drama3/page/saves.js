// A project owns its edits, timers and in-flight requests, even after navigation.
export function createSaves({send, changed = () => {}, saved = () => {}, failed = () => {}}) {
  const projects = new Map();
  function project(name) {
    if (!projects.has(name)) projects.set(name, {channels: new Map(), running: null, revision: 0});
    return projects.get(name);
  }
  function channel(name, key) {
    const channels = project(name).channels;
    if (!channels.has(key)) channels.set(key, {pending: {}, active: {}, timer: null, error: null});
    return channels.get(key);
  }
  function edits(name) {
    return Object.fromEntries([...project(name).channels].map(([key, item]) =>
      [key, {...item.active, ...item.pending}]));
  }
  function status(name) {
    const items = [...project(name).channels.values()];
    if (items.some((item) => item.error)) return "failed";
    if (project(name).running) return "saving";
    return items.some((item) => Object.keys(item.pending).length) ? "pending" : "saved";
  }
  function queue(name, key, body, delay = 700) {
    const item = channel(name, key);
    item.pending = {...item.pending, ...body};
    item.error = null;
    clearTimeout(item.timer);
    item.timer = setTimeout(() => flush(name).catch(() => {}), delay);
    changed(name);
  }
  function flush(name) {
    const group = project(name);
    for (const item of group.channels.values()) { clearTimeout(item.timer); item.timer = null; }
    if (group.running) return group.running;
    // Defer execution so running is installed before callbacks or immediate failures.
    group.running = Promise.resolve().then(async () => {
      while (true) {
        const next = [...group.channels].find(([, item]) => Object.keys(item.pending).length);
        if (!next) return;
        const [key, item] = next;
        item.active = item.pending;
        item.pending = {};
        item.error = null;
        changed(name);
        try {
          const answer = await send(name, key, item.active);
          group.revision++;
          item.active = {};
          saved(name, key, answer);
        } catch (error) {
          item.pending = {...item.active, ...item.pending};
          item.active = {};
          item.error = error;
          failed(name, error);
          throw error;
        }
      }
    }).finally(() => { group.running = null; changed(name); });
    changed(name);
    return group.running;
  }
  return {queue, flush, edits, status, revision: (name) => project(name).revision};
}
