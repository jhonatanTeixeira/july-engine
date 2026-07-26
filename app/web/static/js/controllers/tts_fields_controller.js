(function () {
  // Shows/hides the temperature/semitones/language fields, and swaps in the
  // right per-engine "Idioma" <select>, based on which TTS engine the
  // selected "Modelo" resolves to — mirrors the prefix matching in
  // app/adapters/tts_adapter.py's _ALIAS_ENGINE_MAP (kept here as an exact
  // map since "Modelo" is now a closed <select> over SERVICES_METADATA's
  // fixed ids, not free text).
  const ENGINE_MAP = {
    kokoro: "kokoro",
    chatterbox: "chatterbox",
    "qwen3-tts": "qwen3",
    xtts: "xtts2",
    piper: "piper",
    neutts: "neutts_air",
    indextts: "indextts2",
    "f5-tts": "f5tts",
  };

  class TtsFieldsController extends Stimulus.Controller {
    static targets = ["voiceField", "temperatureField", "semitonesField", "languageField", "languageSelect"];

    connect() {
      this.modelSelect = this.element.querySelector('[name="model"]');
      if (this.modelSelect) {
        this.modelSelect.addEventListener("change", () => this.refresh());
      }
      this.refresh();
    }

    refresh() {
      const modelId = this.modelSelect ? this.modelSelect.value : "";
      const engine = ENGINE_MAP[modelId] || modelId;

      const hasSemitones = engine === "kokoro" || engine === "chatterbox";
      const hasTemperature = ["xtts2", "chatterbox", "qwen3", "neutts_air", "f5tts"].includes(engine);

      if (this.hasTemperatureFieldTarget) {
        this.temperatureFieldTarget.style.display = hasTemperature ? "" : "none";
      }
      if (this.hasSemitonesFieldTarget) {
        this.semitonesFieldTarget.style.display = hasSemitones ? "" : "none";
      }

      let hasLanguage = false;
      this.languageSelectTargets.forEach((select) => {
        const isActive = select.dataset.engine === engine;
        select.hidden = !isActive;
        select.disabled = !isActive;
        if (isActive) hasLanguage = true;
      });
      if (this.hasLanguageFieldTarget) {
        this.languageFieldTarget.style.display = hasLanguage ? "" : "none";
      }
    }
  }

  window.AdminStimulus.register("tts-fields", TtsFieldsController);
})();
