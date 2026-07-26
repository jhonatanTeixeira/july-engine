(function () {
  // A text input styled/behaving like a <select> (arrow, full-list-on-open,
  // click-to-pick) but with type-to-filter — for fields like TTS "Voz" whose
  // valid options are a per-engine subset of a larger catalog (too many to
  // reasonably list in a plain <select>, and native <input list="datalist">
  // narrows its suggestions to only the current exact match once typed,
  // which is the opposite of what a combobox should do).
  //
  // Catalog entries look like {value, label, engines: [...]}; `engines` is
  // matched against the current value of the sibling <select name="model">
  // (mapped id → engine via the engineMap value) so only options valid for
  // the selected TTS engine are ever shown.
  class ComboboxController extends Stimulus.Controller {
    static targets = ["input", "hidden", "list", "catalogData", "engineMapData"];

    connect() {
      this.activeIndex = -1;
      this.catalog = JSON.parse(this.catalogDataTarget.textContent || "[]");
      this.engineMap = JSON.parse(this.engineMapDataTarget.textContent || "{}");
      this.modelSelect = this.element.closest("form")?.querySelector('[name="model"]') || null;
      if (this.modelSelect) {
        this.modelSelect.addEventListener("change", () => this.render());
      }
      this.syncInputFromHidden();
    }

    currentEngine() {
      const modelId = this.modelSelect ? this.modelSelect.value : "";
      return this.engineMap[modelId] || modelId;
    }

    syncInputFromHidden() {
      const current = this.hiddenTarget.value;
      if (!current) return;
      const match = this.catalog.find((o) => o.value === current);
      this.inputTarget.value = match ? match.label : current;
    }

    matches() {
      const engine = this.currentEngine();
      const query = this.inputTarget.value.trim().toLowerCase();
      return this.catalog.filter((o) => {
        if (!o.engines.includes(engine)) return false;
        if (!query) return true;
        return o.value.toLowerCase().includes(query) || o.label.toLowerCase().includes(query);
      });
    }

    render() {
      const items = this.matches();
      this.activeIndex = -1;
      this.listTarget.innerHTML = items
        .map((o, i) => `<li data-index="${i}" data-value="${this._escape(o.value)}">${this._escape(o.label)}</li>`)
        .join("") || `<li class="combo-empty">Nenhuma opção</li>`;
      this._items = items;
    }

    _escape(s) {
      const div = document.createElement("div");
      div.textContent = s;
      return div.innerHTML;
    }

    open() {
      this.render();
      this.listTarget.hidden = false;
    }

    close() {
      this.listTarget.hidden = true;
      this.activeIndex = -1;
    }

    onInput() {
      this.hiddenTarget.value = this.inputTarget.value;
      this.open();
    }

    onFocus() {
      this.open();
    }

    onBlur() {
      // Delay so a click on a list item (which itself blurs the input) has
      // a chance to register before the list disappears.
      this._blurTimer = window.setTimeout(() => this.close(), 150);
    }

    onListMouseDown(event) {
      window.clearTimeout(this._blurTimer);
      const li = event.target.closest("li[data-value]");
      if (!li) return;
      this.select(li.dataset.value);
    }

    select(value) {
      const match = this.catalog.find((o) => o.value === value);
      this.inputTarget.value = match ? match.label : value;
      this.hiddenTarget.value = value;
      this.close();
    }

    onKeydown(event) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        if (this.listTarget.hidden) return this.open();
        this.moveActive(1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        this.moveActive(-1);
      } else if (event.key === "Enter") {
        if (this.activeIndex >= 0 && this._items[this.activeIndex]) {
          event.preventDefault();
          this.select(this._items[this.activeIndex].value);
        }
      } else if (event.key === "Escape") {
        this.close();
      }
    }

    moveActive(delta) {
      if (!this._items || !this._items.length) return;
      this.activeIndex = (this.activeIndex + delta + this._items.length) % this._items.length;
      Array.from(this.listTarget.children).forEach((li, i) => {
        li.classList.toggle("active", i === this.activeIndex);
      });
      const activeLi = this.listTarget.children[this.activeIndex];
      if (activeLi) activeLi.scrollIntoView({ block: "nearest" });
    }
  }

  window.AdminStimulus.register("combobox", ComboboxController);
})();
