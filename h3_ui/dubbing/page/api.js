// The page's calls to /dubbing/api. Every refusal arrives as {"error": message} and is thrown.

const BASE = "api";

async function call(method, path, body, query) {
  const url = new URL(`${BASE}/${path}`, location.href);
  for (const [key, value] of Object.entries(query || {})) {
    if (value !== undefined && value !== null) url.searchParams.set(key, value);
  }
  const options = {method, credentials: "same-origin", headers: {}};
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let data = null;
  try {
    data = await response.json();
  } catch (_) {
    data = null;
  }
  if (!response.ok || (data && data.error)) {
    throw new Error((data && (data.error || data.detail)) || `Request failed (${response.status}).`);
  }
  return data;
}

export const api = {
  get: (path, query) => call("GET", path, undefined, query),
  post: (path, body = {}) => call("POST", path, body),
  patch: (path, body) => call("PATCH", path, body),
  put: (path, body) => call("PUT", path, body),
  del: (path, query) => call("DELETE", path, undefined, query),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return call("POST", "upload", form);
  },
  url: (path, query) => {
    const url = new URL(`${BASE}/${path}`, location.href);
    for (const [key, value] of Object.entries(query || {})) url.searchParams.set(key, value);
    return url.pathname + url.search;
  },
};
