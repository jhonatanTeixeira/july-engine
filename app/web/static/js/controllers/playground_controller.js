(function () {
  // "Testar Modelos" chat playground. Fully client-side conversation state —
  // POST /admin/playground/send is stateless, every field it needs (model,
  // full messages array, settings) travels in the request. Model switch,
  // regenerate, and delete are pure client-side array/DOM operations, no
  // server round-trip for any of them (send is the only network call).
  const ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024;

  class PlaygroundController extends Stimulus.Controller {
    static targets = [
      "modelSelect", "messagesList", "composerForm", "attachmentTray", "fileInput", "textInput",
      "settingsForm", "settingsBody", "chevron", "temperatureNumber", "temperatureRange", "modelsData",
      "presetsData",
    ];

    connect() {
      this.models = JSON.parse(this.modelsDataTarget.textContent || "[]");
      this.presets = JSON.parse(this.presetsDataTarget.textContent || "[]");
      this.sessions = {}; // alias -> { messages: [{role, content, html}], lastPromptTokens }
      this.pendingAttachments = [];
      this.currentAlias = this.modelSelectTarget.value;
      this._ensureSession(this.currentAlias);
      this._renderSession(this.currentAlias);
    }

    // ------------------------------------------------------------------
    // Session / model switching
    // ------------------------------------------------------------------

    _ensureSession(alias) {
      if (!this.sessions[alias]) {
        this.sessions[alias] = { messages: [], lastPromptTokens: null };
      }
      return this.sessions[alias];
    }

    // `alias` here is a PRESET alias (the dropdown's value) — resolve it to the
    // underlying model_alias the preset points at, then look up that model's real
    // capability fields (context_window, etc.). Falls back to {} if the preset's
    // `model` doesn't match anything in the catalog (e.g. a misconfigured preset).
    _modelInfo(alias) {
      const preset = this.presets.find((p) => p.alias === alias);
      if (!preset) return {};
      return this.models.find((m) => m.model_alias === preset.model) || {};
    }

    switchModel() {
      this.currentAlias = this.modelSelectTarget.value;
      this._ensureSession(this.currentAlias);
      this._renderSession(this.currentAlias);
    }

    _renderSession(alias) {
      const session = this.sessions[alias];
      this.messagesListTarget.innerHTML = session.messages.map((m) => m.html).join("");
      this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;
    }

    _domIndexToArrayIndex(session, domIndex) {
      const hasSystem = session.messages.length > 0 && session.messages[0].role === "system";
      return domIndex + (hasSystem ? 1 : 0);
    }

    // ------------------------------------------------------------------
    // Settings panel
    // ------------------------------------------------------------------

    toggleSettings() {
      const collapsed = this.settingsBodyTarget.style.display === "none";
      this.settingsBodyTarget.style.display = collapsed ? "" : "none";
      this.chevronTarget.textContent = collapsed ? "▲" : "▼";
    }

    syncTempFromSlider() {
      this.temperatureNumberTarget.value = this.temperatureRangeTarget.value;
    }

    syncTempFromNumber() {
      this.temperatureRangeTarget.value = this.temperatureNumberTarget.value;
    }

    _readSettings() {
      const form = this.settingsFormTarget;
      const num = (name) => {
        const v = form.elements[name].value;
        return v === "" ? null : Number(v);
      };
      const str = (name) => {
        const v = form.elements[name].value.trim();
        return v === "" ? null : v;
      };

      let stop = null;
      const stopRaw = str("stop");
      if (stopRaw) stop = stopRaw.split("\n").map((s) => s.trim()).filter(Boolean);

      let responseFormat = null;
      const rfRaw = str("response_format");
      if (rfRaw) {
        try { responseFormat = JSON.parse(rfRaw); } catch (e) { console.warn("playground: invalid response_format JSON", e); }
      }

      let tools = null;
      const toolsRaw = str("tools");
      if (toolsRaw) {
        try { tools = JSON.parse(toolsRaw); } catch (e) { console.warn("playground: invalid tools JSON", e); }
      }

      return {
        stream: form.elements["stream"].checked,
        max_tokens: num("max_tokens"),
        max_completion_tokens: num("max_completion_tokens"),
        temperature: num("temperature"),
        top_p: num("top_p"),
        top_k: num("top_k"),
        min_p: num("min_p"),
        repetition_penalty: num("repetition_penalty"),
        stop,
        response_format: responseFormat,
        tools,
      };
    }

    _systemPrompt() {
      return this.settingsFormTarget.elements["system_prompt"].value;
    }

    // ------------------------------------------------------------------
    // Attachments
    // ------------------------------------------------------------------

    openFilePicker() {
      this.fileInputTarget.click();
    }

    addFiles(event) {
      const files = Array.from(event.target.files || []);
      files.forEach((file) => this._addAttachment(file));
      this.fileInputTarget.value = "";
    }

    _addAttachment(file) {
      if (file.size > ATTACHMENT_MAX_BYTES) {
        this._flashAttachmentError(`${file.name}: arquivo maior que 20MB`);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        this.pendingAttachments.push({
          id: `att-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          name: file.name,
          mime: file.type,
          dataUrl: reader.result,
        });
        this._renderAttachmentTray();
      };
      reader.onerror = () => this._flashAttachmentError(`${file.name}: falha ao ler arquivo`);
      reader.readAsDataURL(file);
    }

    _flashAttachmentError(message) {
      const chip = document.createElement("span");
      chip.className = "attachment-chip attachment-error";
      chip.textContent = message;
      this.attachmentTrayTarget.appendChild(chip);
      setTimeout(() => chip.remove(), 4000);
    }

    _renderAttachmentTray() {
      this.attachmentTrayTarget.innerHTML = this.pendingAttachments
        .map((a) => `<span class="attachment-chip" data-id="${a.id}">${this._escape(a.name)} <button type="button" data-action="click->playground#removeAttachment" data-id="${a.id}">×</button></span>`)
        .join("");
    }

    removeAttachment(event) {
      const id = event.target.closest("[data-id]").dataset.id;
      this.pendingAttachments = this.pendingAttachments.filter((a) => a.id !== id);
      this._renderAttachmentTray();
    }

    _contentPartsFromAttachments() {
      return this.pendingAttachments.map((a) => {
        if (a.mime.startsWith("image/")) return { type: "image_url", image_url: { url: a.dataUrl } };
        if (a.mime.startsWith("audio/")) return { type: "audio_url", audio_url: { url: a.dataUrl } };
        if (a.mime.startsWith("video/")) return { type: "video_url", video_url: { url: a.dataUrl } };
        if (a.mime === "application/pdf") return { type: "file_url", file_url: { url: a.dataUrl, media_type: "application/pdf" } };
        return { type: "text", text: `[anexo não suportado: ${a.name}]` };
      });
    }

    _escape(s) {
      const div = document.createElement("div");
      div.textContent = s;
      return div.innerHTML;
    }

    // Renders raw model/user text as sanitized markdown HTML. The source text is
    // HTML-escaped BEFORE hitting marked, so any literal `<tag>` in model output
    // displays as text instead of being parsed as markup; the resulting HTML (only
    // markup marked itself generated from markdown syntax) is then swept for
    // script/style/event-handler/javascript: injection before being used.
    _renderMarkdown(rawText) {
      if (!rawText) return "";
      if (!window.marked) return this._escape(rawText);
      let html;
      try {
        html = window.marked.parse(this._escape(rawText), { breaks: true, gfm: true });
      } catch (e) {
        return this._escape(rawText);
      }
      return this._sanitizeHtml(html);
    }

    _sanitizeHtml(html) {
      const template = document.createElement("template");
      template.innerHTML = html;
      template.content.querySelectorAll("script, style, iframe, object, embed, link, meta, form").forEach((el) => el.remove());
      template.content.querySelectorAll("*").forEach((el) => {
        Array.from(el.attributes).forEach((attr) => {
          const name = attr.name.toLowerCase();
          const value = attr.value.trim();
          if (name.startsWith("on")) {
            el.removeAttribute(attr.name);
          } else if ((name === "href" || name === "src") && /^\s*(javascript:|vbscript:|data:text\/html)/i.test(value)) {
            el.removeAttribute(attr.name);
          }
        });
      });
      return template.innerHTML;
    }

    // Reads the raw text a server-rendered bubble carries in .msg-content, then
    // replaces it with the rendered markdown — used for fragments that arrive as
    // plain escaped text (non-streaming responses), where there's no client-side
    // raw-text accumulator to render from directly.
    _finalizeAssistantBubble(el) {
      const contentEl = el.querySelector(".msg-content");
      const raw = contentEl.textContent;
      contentEl.innerHTML = this._renderMarkdown(raw);
      return raw;
    }

    // ------------------------------------------------------------------
    // Composer / send
    // ------------------------------------------------------------------

    onComposerKeydown(event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.send(event);
      }
    }

    async send(event) {
      event.preventDefault();
      const text = this.textInputTarget.value.trim();
      if (!text && this.pendingAttachments.length === 0) return;

      const parts = this._contentPartsFromAttachments();
      if (text) parts.push({ type: "text", text });
      const userContent = parts.length === 1 && parts[0].type === "text" ? text : parts;

      const session = this._ensureSession(this.currentAlias);
      this._syncSystemMessage(session);

      const userHtml = this._renderUserBubble(text, this.pendingAttachments);
      session.messages.push({ role: "user", content: userContent, html: userHtml });
      this.messagesListTarget.insertAdjacentHTML("beforeend", userHtml);
      this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;

      this.textInputTarget.value = "";
      this.pendingAttachments = [];
      this._renderAttachmentTray();

      this._maybeTrim(session);
      await this._generate(session);
    }

    _syncSystemMessage(session) {
      const systemPrompt = this._systemPrompt();
      if (session.messages.length === 0) {
        session.messages.push({ role: "system", content: systemPrompt, html: "" });
      } else if (session.messages[0].role === "system") {
        session.messages[0].content = systemPrompt;
      } else {
        session.messages.unshift({ role: "system", content: systemPrompt, html: "" });
      }
    }

    _renderUserBubble(text, attachments) {
      const chips = attachments.map((a) => `<span class="attachment-chip">${this._escape(a.name)}</span>`).join("");
      return (
        `<div class="msg-bubble msg-user">` +
        `<div class="msg-header"><span class="msg-avatar">🧑</span><span class="msg-name">You</span></div>` +
        `<div class="msg-content">${this._renderMarkdown(text)}</div>` +
        (chips ? `<div class="msg-footer">${chips}</div>` : "") +
        `<div class="msg-actions"><button type="button" class="msg-icon-btn" data-action="click->playground#deleteMessage" title="Excluir">🗑</button></div>` +
        `</div>`
      );
    }

    async _generate(session) {
      const settings = this._readSettings();
      const body = {
        model: this.currentAlias,
        messages: session.messages.map((m) => ({ role: m.role, content: m.content })),
        ...settings,
      };

      const placeholder = document.createElement("div");
      placeholder.className = "msg-bubble msg-assistant msg-pending";
      placeholder.innerHTML =
        `<div class="msg-header"><span class="msg-avatar">🤖</span><span class="msg-name">${this._escape(this.currentAlias)}</span></div>` +
        `<details class="msg-reasoning" hidden><summary>Reasoning</summary><div class="msg-reasoning-content"></div></details>` +
        `<div class="msg-content"><span class="msg-typing">●●●</span></div>`;
      this.messagesListTarget.appendChild(placeholder);
      this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;

      let response;
      try {
        response = await fetch("/admin/playground/send", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        placeholder.querySelector(".msg-content").textContent = `Erro de rede: ${e}`;
        return;
      }

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        placeholder.querySelector(".msg-content").textContent = `Erro: ${text || response.status}`;
        return;
      }

      if (settings.stream) {
        await this._consumeStream(response, placeholder, session);
      } else {
        const html = await response.text();
        placeholder.replaceWith(this._firstElement(html));
        const lastEl = this.messagesListTarget.lastElementChild;
        const content = this._finalizeAssistantBubble(lastEl);
        const promptTokens = lastEl.dataset.promptTokens ? Number(lastEl.dataset.promptTokens) : null;
        session.messages.push({ role: "assistant", content, html: lastEl.outerHTML });
        if (promptTokens != null) session.lastPromptTokens = promptTokens;
      }
      this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;
    }

    _firstElement(html) {
      const tmp = document.createElement("div");
      tmp.innerHTML = html.trim();
      return tmp.firstElementChild;
    }

    async _consumeStream(response, placeholder, session) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const contentEl = placeholder.querySelector(".msg-content");
      const reasoningEl = placeholder.querySelector(".msg-reasoning");
      const reasoningContentEl = placeholder.querySelector(".msg-reasoning-content");
      let buffer = "";
      let firstDelta = true;
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop();

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          let payload;
          try {
            payload = JSON.parse(line.slice(5).trim());
          } catch (e) {
            continue;
          }

          if (payload.error) {
            contentEl.textContent = `Erro: ${payload.error}`;
          } else if (payload.reasoning_delta) {
            reasoningEl.hidden = false;
            reasoningContentEl.textContent += payload.reasoning_delta;
            this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;
          } else if (payload.delta) {
            if (firstDelta) {
              contentEl.textContent = "";
              firstDelta = false;
            }
            // fullText is the raw accumulator — rendered progressively as markdown
            // for live display, but kept as plain text for conversation history.
            fullText += payload.delta;
            contentEl.innerHTML = this._renderMarkdown(fullText);
            this.messagesListTarget.scrollTop = this.messagesListTarget.scrollHeight;
          } else if (payload.done) {
            placeholder.replaceWith(this._firstElement(payload.html));
            const lastEl = this.messagesListTarget.lastElementChild;
            lastEl.querySelector(".msg-content").innerHTML = this._renderMarkdown(fullText);
            session.messages.push({ role: "assistant", content: fullText, html: lastEl.outerHTML });
            if (payload.metrics && payload.metrics.prompt_tokens != null) {
              session.lastPromptTokens = payload.metrics.prompt_tokens;
            }
          }
        }
      }
    }

    // ------------------------------------------------------------------
    // Token-budget trimming — client-side, since conversation state lives here
    // ------------------------------------------------------------------

    // Media parts (image_url/audio_url/video_url/file_url) carry a base64 data URL —
    // its length reflects the attachment's file size, not anything close to its real
    // token cost (the model's own encoder turns an image into a small fixed number
    // of tokens, regardless of how many base64 characters represent it). Counting
    // that raw length as if it were text wildly inflates the estimate, so only
    // actual text contributes to the character-based heuristic; each media part
    // gets a flat placeholder cost instead.
    _estimateTokens(content) {
      if (typeof content === "string") return Math.ceil(content.length / 4);
      if (!Array.isArray(content)) return Math.ceil(JSON.stringify(content).length / 4);

      const MEDIA_TOKEN_ESTIMATE = 256;
      return content.reduce((sum, part) => {
        if (part.type === "text") return sum + Math.ceil((part.text || "").length / 4);
        if (part.type === "image_url" || part.type === "audio_url" || part.type === "video_url" || part.type === "file_url") {
          return sum + MEDIA_TOKEN_ESTIMATE;
        }
        return sum + Math.ceil(JSON.stringify(part).length / 4);
      }, 0);
    }

    _maybeTrim(session) {
      // Nothing real to judge against yet (no completed turn in this session) — and
      // a brand-new conversation has no accumulated history worth trimming anyway.
      // Deliberately NOT falling back to a character-count estimate here: that
      // heuristic has no reliable way to price attachments (a data URL's length
      // reflects file size, not token cost), and guessing wrong on this exact call
      // is what caused it to delete the message that triggered it in the first
      // place. Once a real turn completes, session.lastPromptTokens is the only
      // number driving this from then on.
      if (session.lastPromptTokens == null) return;

      const model = this._modelInfo(this.currentAlias);
      const contextWindow = model.context_window || 4096;
      const total = session.lastPromptTokens;
      if (total <= contextWindow * 0.9) return;

      const hasSystem = session.messages.length > 0 && session.messages[0].role === "system";
      const keep = hasSystem ? session.messages.slice(0, 1) : [];
      // The last message in `rest` is always the one that just triggered this trim
      // (the turn currently being sent) — it must never be the one dropped to make
      // room for itself, so it's excluded from what the loop below is allowed to eat.
      const rest = hasSystem ? session.messages.slice(1) : session.messages.slice();
      const protectedTail = rest.length > 0 ? 1 : 0;
      const trimmable = rest.slice(0, rest.length - protectedTail);

      const target = total * 0.3;
      let dropped = 0;
      let idx = 0;
      while (idx < trimmable.length && dropped < target) {
        dropped += this._estimateTokens(trimmable[idx].content);
        idx += 1;
      }

      session.messages = keep.concat(rest.slice(idx));
      session.lastPromptTokens = null; // stale after trimming — re-derive via heuristic until the next real usage
      this._renderSession(this.currentAlias);
    }

    // ------------------------------------------------------------------
    // Regenerate / delete — pure client-side, no request for delete
    // ------------------------------------------------------------------

    regenerate(event) {
      const session = this.sessions[this.currentAlias];
      const bubble = event.target.closest(".msg-bubble");
      const domIndex = Array.from(this.messagesListTarget.children).indexOf(bubble);
      if (domIndex < 0) return;

      const arrIndex = this._domIndexToArrayIndex(session, domIndex);
      session.messages = session.messages.slice(0, arrIndex);
      this._renderSession(this.currentAlias);
      this._generate(session);
    }

    deleteMessage(event) {
      const session = this.sessions[this.currentAlias];
      const bubble = event.target.closest(".msg-bubble");
      const domIndex = Array.from(this.messagesListTarget.children).indexOf(bubble);
      if (domIndex < 0) return;

      const arrIndex = this._domIndexToArrayIndex(session, domIndex);
      session.messages.splice(arrIndex, 1);
      this._renderSession(this.currentAlias);
    }
  }

  window.AdminStimulus.register("playground", PlaygroundController);
})();
