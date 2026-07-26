(function () {
  // Streams the SSE-style response from POST /admin/models/download (new
  // model, full form) or POST /admin/models/{alias}/download (existing
  // model, no body — just re-ensures its already-saved files are on disk)
  // via a real fetch() (not EventSource, since the route is POST), updating
  // a progress bar, and refreshing the models grid + closing the modal on
  // success. The underlying generator only reports coarse stages
  // (initializing/starting/downloading/success/error), not byte-level
  // progress, so the bar reflects stage, not real percentage.
  const STAGE_PERCENT = {
    initializing: 10,
    starting: 20,
    downloading: 60,
    success: 100,
  };

  class DownloadProgressController extends Stimulus.Controller {
    static targets = ["form", "status"];

    // "Instalar": registers a new model (or overwrites an existing one) from
    // the full form, then downloads it.
    async start(event) {
      event.preventDefault();
      // This is a plain button, not a submit, so the browser never runs its
      // native constraint validation (pattern/required) on its own — trigger
      // it explicitly so e.g. a "/" in the alias is caught before the request.
      if (!this.formTarget.reportValidity()) return;
      await this._stream("/admin/models/download", new FormData(this.formTarget));
    }

    // "Baixar": no form data sent at all — just (re)downloads whatever is
    // already saved for this model, without touching its metadata.
    async download(event) {
      event.preventDefault();
      await this._stream(event.params.url, null);
    }

    async _stream(url, body) {
      this.setStatus("Iniciando...", STAGE_PERCENT.initializing);

      let response;
      try {
        response = await fetch(url, { method: "POST", body });
      } catch (e) {
        this.setStatus(`Erro de rede: ${e}`, null, true);
        return;
      }

      if (!response.ok || !response.body) {
        const text = await response.text().catch(() => "");
        this.setStatus(`Erro: ${text || response.status}`, null, true);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop(); // last chunk may be incomplete, keep for next read

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          try {
            this.handleEvent(JSON.parse(line.slice(5).trim()));
          } catch (e) {
            // ignore malformed frame
          }
        }
      }
    }

    handleEvent(payload) {
      const status = payload.status;
      const message = payload.message || "";

      if (status === "error") {
        this.setStatus(`Erro: ${message}`, null, true);
        return;
      }

      if (status === "success") {
        this.setStatus(message || "Download concluído!", STAGE_PERCENT.success);
        setTimeout(() => {
          // Full-tab refresh also clears the modal (tabs/models.html always
          // renders an empty #modal-container).
          window.htmx.ajax("GET", "/admin/tab/models", { target: "#tab-content", swap: "innerHTML" });
        }, 600);
        return;
      }

      this.setStatus(message, STAGE_PERCENT[status] || null);
    }

    setStatus(message, percent, isError) {
      if (!this.hasStatusTarget) return;
      const color = isError ? "var(--danger)" : "var(--text-secondary)";
      let html = `<p style="font-size:12px; color:${color};">${message}</p>`;
      if (percent !== null && percent !== undefined) {
        html += `<div class="progress-bar"><div class="progress-bar-fill" style="width:${percent}%;"></div></div>`;
      }
      this.statusTarget.innerHTML = html;
    }
  }

  window.AdminStimulus.register("download-progress", DownloadProgressController);
})();
