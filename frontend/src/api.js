const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

export function createSession(complaint) {
  return request("/v1/sessions", {
    method: "POST",
    body: JSON.stringify({ complaint }),
  });
}

export function approveSession(id) {
  return request(`/v1/sessions/${id}/approve`, { method: "POST" });
}

export function previewMail(id, to) {
  return request(`/v1/sessions/${id}/mail/preview`, {
    method: "POST",
    body: JSON.stringify({ to }),
  });
}

export function sendMail(id, to) {
  return request(`/v1/sessions/${id}/mail/send`, {
    method: "POST",
    body: JSON.stringify({ to }),
  });
}
