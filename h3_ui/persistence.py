"""Small browser-side helpers for remembering non-sensitive UI settings."""

from __future__ import annotations


CLIENT_SETTINGS_PERSISTENCE_JS = r"""() => {
    const storageKey = "minimax-h3:ui-selections:v1";
    const seen = new WeakSet();
    let values = {};
    try { values = JSON.parse(window.localStorage.getItem(storageKey) || "{}"); }
    catch (_) { values = {}; }

    const componentFor = (control) =>
        control.closest(".block, [data-testid='block']") || control.parentElement;
    const keyFor = (control) => {
        const component = componentFor(control);
        const label = component?.querySelector("label")?.innerText?.trim();
        return `${component?.id || label || control.type}:${label || ""}`;
    };
    const controls = () => Array.from(document.querySelectorAll(
        ".gradio-container select, .gradio-container input[type='checkbox'], " +
        ".gradio-container input[type='radio'], .gradio-container input[type='range'], " +
        ".gradio-container input[type='number'], .gradio-container input[aria-haspopup='listbox'], " +
        ".gradio-container [role='combobox'], .gradio-container [role='radio']"));
    const write = () => {
        try { window.localStorage.setItem(storageKey, JSON.stringify(values)); }
        catch (_) { /* Private browsing or full storage must not break the UI. */ }
    };
    const restore = (control) => {
        if (seen.has(control)) return;
        seen.add(control);
        const key = keyFor(control);
        if (!Object.prototype.hasOwnProperty.call(values, key)) return;
        const stored = values[key];
        if (control.type === "radio") control.checked = control.value === String(stored);
        else if (control.getAttribute("role") === "radio") {
            if ((control.innerText.trim() || control.getAttribute("aria-label")) === String(stored)) control.click();
        } else if (control.type === "checkbox") control.checked = Boolean(stored);
        else control.value = String(stored);
        control.dispatchEvent(new Event("input", { bubbles: true }));
        control.dispatchEvent(new Event("change", { bubbles: true }));
    };
    const remember = (event) => {
        const control = event.target;
        const role = control.getAttribute("role");
        if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement || role === "combobox" || role === "radio")) return;
        if (role === "radio") {
            if (control.getAttribute("aria-checked") !== "true") return;
            values[keyFor(control)] = control.innerText.trim() || control.getAttribute("aria-label") || "";
            write();
            return;
        }
        if (!["checkbox", "radio", "range", "number", "select-one", "select-multiple", "text"].includes(control.type) && role !== "combobox") return;
        if (control.type === "radio" && !control.checked) return;
        values[keyFor(control)] = control.type === "checkbox" ? control.checked : control.value;
        write();
    };
    document.addEventListener("input", remember, true);
    document.addEventListener("change", remember, true);
    const restoreAll = () => controls().forEach(restore);
    restoreAll();
    window.setTimeout(restoreAll, 250);
    new MutationObserver(restoreAll).observe(document.body, { childList: true, subtree: true });
};
"""
