(function () {
  // GGUF add/edit modal behavior: toggling the vision fields, refreshing the
  // filename <select> from HF when the repo changes, and auto-filling
  // template/vision/etc. via /admin/models/detect once a file is picked.
  // (Kept as real JS rather than declarative HTMX because it's cross-field
  // logic — one field's change drives fetching + populating others.)
  class ModelFormController extends Stimulus.Controller {
    static targets = [
      "form", "modelId", "filename", "template", "visionToggle", "audioToggle", "mmprojFields",
      "mmprojId", "mmprojFilename", "cpuMoeToggle", "nCpuMoe", "moeFields", "mtpFields",
    ];

    connect() {
      this.toggleMmprojFields();
      this.cpuMoeChanged();
      // Only auto-fill the mmproj repo/filename from the main model's file list
      // until the user directly edits the mmproj repo id themselves — from then
      // on, "modelIdChanged" backs off and leaves their choice alone.
      this._mmprojManuallySet = false;

      // Clicking a HF search result opens this modal with model_id already
      // server-rendered (prefill_repo_id), which never fires a DOM "change"
      // event — so the filename <select> would stay empty until the user
      // touched the field themselves. Editing an existing model already ships
      // its current filename as a pre-selected <option>, so only backfill
      // when the list is still empty (the "new model with prefilled repo id"
      // case).
      if (this.hasModelIdTarget && this.modelIdTarget.value.trim() &&
          this.hasFilenameTarget && this.filenameTarget.options.length === 0) {
        this.modelIdChanged();
      }
    }

    // Called on htmx:afterSwap of #vram-estimate-container — that response
    // already carries is_moe (computed from the real GGUF tensor metadata),
    // so it's the one place that reliably knows MoE-ness for both "editing an
    // existing model" (fires as soon as the modal opens) and "adding a new
    // one" (fires once a file is picked). Only toggles visibility — never
    // touches the fields' own values, so mid-edit state is never clobbered.
    syncMoeFields(event) {
      if (!this.hasMoeFieldsTarget) return;
      const card = event.target.querySelector(".vram-card");
      const isMoe = !!card && card.dataset.isMoe === "true";
      this.moeFieldsTarget.style.display = isMoe ? "" : "none";
    }

    // Same idea as syncMoeFields, for has_mtp (computed from the GGUF's
    // nextn_predict_layers header / NextN tensor names) — only appears for
    // models with trained MTP heads to load/use.
    syncMtpFields(event) {
      if (!this.hasMtpFieldsTarget) return;
      const card = event.target.querySelector(".vram-card");
      const hasMtp = !!card && card.dataset.hasMtp === "true";
      this.mtpFieldsTarget.style.display = hasMtp ? "" : "none";
    }

    toggleMmprojFields() {
      if (!this.hasMmprojFieldsTarget) return;
      const visionOn = this.hasVisionToggleTarget && this.visionToggleTarget.checked;
      const audioOn = this.hasAudioToggleTarget && this.audioToggleTarget.checked;
      this.mmprojFieldsTarget.style.display = (visionOn || audioOn) ? "" : "none";
    }

    cpuMoeChanged() {
      if (!this.hasCpuMoeToggleTarget || !this.hasNCpuMoeTarget) return;
      // n_cpu_moe is redundant once cpu_moe=True (llama-cpp-python already
      // keeps every MoE layer on CPU in that case) — disable it to avoid
      // suggesting it still does something.
      this.nCpuMoeTarget.disabled = this.cpuMoeToggleTarget.checked;
    }

    async modelIdChanged() {
      const repoId = this.modelIdTarget.value.trim();
      if (!repoId || !this.hasFilenameTarget) return;

      let html;
      try {
        const res = await fetch(`/admin/models/files?repo_id=${encodeURIComponent(repoId)}`);
        if (!res.ok) return;
        html = await res.text();
      } catch (e) {
        console.warn("model-form: failed to refresh file list", e);
        return;
      }

      this.filenameTarget.innerHTML = html;

      // mmproj files are themselves .gguf files, so the same per-repo file list
      // already contains them — reuse it instead of a second fetch, unless the
      // user has taken over the mmproj repo field manually.
      if (this._mmprojManuallySet || !this.hasMmprojFilenameTarget) return;

      this.mmprojFilenameTarget.innerHTML = html;
      const mmprojMatch = Array.from(this.mmprojFilenameTarget.options)
        .map((o) => o.value)
        .find((v) => v.toLowerCase().includes("mmproj"));

      if (mmprojMatch) {
        this.mmprojFilenameTarget.value = mmprojMatch;
        if (this.hasMmprojIdTarget) this.mmprojIdTarget.value = repoId;
        if (this.hasVisionToggleTarget && !this.visionToggleTarget.checked) {
          this.visionToggleTarget.checked = true;
          this.toggleMmprojFields();
        }
      }
    }

    // Fires only on a real user edit to the mmproj repo id (programmatic
    // `.value =` assignments from modelIdChanged don't trigger "change"), so
    // this is exactly the "user took over" signal that stops further
    // auto-filling from the main model's own file list.
    async mmprojIdChanged() {
      this._mmprojManuallySet = true;
      const repoId = this.mmprojIdTarget.value.trim();
      if (!repoId || !this.hasMmprojFilenameTarget) return;

      try {
        const res = await fetch(`/admin/models/files?repo_id=${encodeURIComponent(repoId)}`);
        if (res.ok) {
          this.mmprojFilenameTarget.innerHTML = await res.text();
        }
      } catch (e) {
        console.warn("model-form: failed to refresh mmproj file list", e);
      }
    }

    async filenameChanged() {
      const repoId = this.hasModelIdTarget ? this.modelIdTarget.value.trim() : "";
      const filename = this.hasFilenameTarget ? this.filenameTarget.value.trim() : "";
      if (!repoId || !filename) return;

      try {
        const res = await fetch("/admin/models/detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: repoId, filename: filename }),
        });
        if (!res.ok) return;
        const data = await res.json();
        const meta = data.detected_metadata || {};

        if (meta.template && this.hasTemplateTarget) {
          this.templateTarget.value = meta.template;
        }
        if (meta.is_vision && this.hasVisionToggleTarget && !this.visionToggleTarget.checked) {
          this.visionToggleTarget.checked = true;
          this.toggleMmprojFields();
        }
      } catch (e) {
        console.warn("model-form: metadata detection failed", e);
      }
    }
  }

  window.AdminStimulus.register("model-form", ModelFormController);
})();
