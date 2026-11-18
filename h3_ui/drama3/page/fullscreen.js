export function bindFullscreen(view, button, reportError, signal) {
  if (!button || !document.fullscreenEnabled || !view?.requestFullscreen) return null;
  const active = () => document.fullscreenElement === view;
  const update = () => {
    button.textContent = active() ? "Exit fullscreen" : "Fullscreen";
  };
  const change = async (enter) => {
    try {
      if (enter) await view.requestFullscreen();
      else if (active()) await document.exitFullscreen();
    } catch {
      reportError("Could not change fullscreen mode.");
    }
  };
  button.hidden = false;
  update();
  view.addEventListener("fullscreenchange", update, {signal});
  button.addEventListener("click", () => change(!active()), {signal});
  return {exit: () => change(false)};
}
