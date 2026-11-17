"use strict";

(async () => {
  const $ = (selector) => document.querySelector(selector);
  const gallery = $("#gallery");
  const previewDialog = $("#preview-dialog");
  const exportDialog = $("#export-dialog");
  const storageKey = "lcisp-scene-picker-v1";
  let scenes;
  try {
    const response = await fetch("scenes.json?v=all999");
    if (!response.ok) throw new Error("scene list unavailable");
    scenes = await response.json();
  } catch {
    $("#loading").textContent = "图片列表加载失败，请刷新重试。";
    return;
  }
  const sceneMap = new Map(scenes.map((scene) => [scene.id, scene]));
  let selected = new Set();
  let hero = null;
  let filter = "all";
  let currentScene = null;
  let visibleScenes = [];

  function normalizeId(value) {
    if (typeof value !== "string" || !/^\d{1,4}$/.test(value)) return null;
    const id = value.padStart(4, "0");
    return sceneMap.has(id) ? id : null;
  }

  function restore(state) {
    if (!state || !Array.isArray(state.scenes)) return;
    selected = new Set(state.scenes.map(normalizeId).filter(Boolean));
    hero = normalizeId(state.hero);
    if (!selected.has(hero)) hero = selected.values().next().value || null;
  }

  function readHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (!params.has("scenes")) return false;
    restore({scenes: params.get("scenes").split(","), hero: params.get("hero")});
    return true;
  }

  if (!readHash()) {
    try { restore(JSON.parse(localStorage.getItem(storageKey))); } catch {}
  }

  function save() {
    const state = {scenes: [...selected], hero};
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch {}
    const params = new URLSearchParams();
    if (selected.size) {
      params.set("scenes", [...selected].join(","));
      params.set("hero", hero);
    }
    history.replaceState(null, "", `${location.pathname}${location.search}${params.size ? `#${params}` : ""}`);
  }

  function updateSlider() {
    const value = Number($("#preview-slider").value);
    const pair = $("#preview-pair");
    pair.style.setProperty("--position", `${value}%`);
    pair.dataset.endpoint = value === 0 ? "reference" : value === 100 ? "raw" : "split";
    $("#preview-slider").setAttribute("aria-valuetext", `${value}% RAW, ${100 - value}% Reference`);
  }

  function updateSelection() {
    $("#selection-count").textContent = `已选 ${selected.size} 张`;
    $("#filter-selected-count").textContent = selected.size;
    $("#hero-summary").textContent = hero ? `右上角：${hero}` : "右上角：未选择";
    $("#clear").disabled = !selected.size;
    $("#export").disabled = !selected.size;
    $("#selected-ids").replaceChildren(...[...selected].map((id) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "selected-chip";
      button.textContent = `${id} ×`;
      button.classList.toggle("hero-chip", id === hero);
      button.setAttribute("aria-label", `取消选择 ${id}`);
      button.addEventListener("click", () => toggle(id, false));
      return button;
    }));
    for (const card of gallery.children) {
      const id = card.dataset.id;
      card.classList.toggle("is-selected", selected.has(id));
      card.classList.toggle("is-hero", id === hero);
      card.querySelector("input").checked = selected.has(id);
      const button = card.querySelector(".hero-pick");
      button.setAttribute("aria-pressed", String(id === hero));
      button.textContent = id === hero ? "右上角" : "用于右上角";
    }
    if (currentScene) {
      $("#preview-select").textContent = selected.has(currentScene) ? "取消选择" : "选择这张";
      $("#preview-select").setAttribute("aria-pressed", String(selected.has(currentScene)));
      $("#preview-hero").textContent = currentScene === hero ? "已用于右上角" : "用于右上角";
      $("#preview-hero").setAttribute("aria-pressed", String(currentScene === hero));
    }
  }

  function toggle(id, checked) {
    if (checked) {
      selected.add(id);
      if (!hero) hero = id;
    } else {
      selected.delete(id);
      if (hero === id) hero = selected.values().next().value || null;
    }
    save();
    if (filter === "selected") render();
    else updateSelection();
  }

  function setHero(id) {
    selected.add(id);
    hero = id;
    save();
    if (filter === "selected") render();
    else updateSelection();
  }

  function cardFor(scene) {
    const card = document.createElement("article");
    card.className = "scene-card";
    card.dataset.id = scene.id;
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "open-preview";
    preview.setAttribute("aria-label", `放大场景 ${scene.id} 并查看 RAW / Reference`);
    const image = document.createElement("img");
    image.src = scene.reference;
    image.alt = `场景 ${scene.id} 完整画面`;
    image.width = 640;
    image.height = 480;
    image.loading = "lazy";
    image.decoding = "async";
    preview.append(image);
    const zoom = document.createElement("span");
    zoom.className = "zoom-icon";
    zoom.setAttribute("aria-hidden", "true");
    zoom.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 3l6 6m12-6-6 6M3 21l6-6m12 6-6-6"/></svg>';
    preview.append(zoom);
    preview.addEventListener("click", () => openPreview(scene.id));
    const label = document.createElement("label");
    label.className = "check-label";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.setAttribute("aria-label", `选择场景 ${scene.id}`);
    checkbox.addEventListener("change", () => toggle(scene.id, checkbox.checked));
    label.append(checkbox, "选择");
    const footer = document.createElement("div");
    footer.className = "card-footer";
    const id = document.createElement("span");
    id.className = "scene-id";
    id.textContent = scene.id;
    const cover = document.createElement("button");
    cover.type = "button";
    cover.className = "hero-pick";
    cover.addEventListener("click", () => setHero(scene.id));
    footer.append(id, cover);
    card.append(preview, label, footer);
    return card;
  }

  function render() {
    const query = $("#search").value.trim();
    visibleScenes = scenes.filter((scene) =>
      (filter === "all" || selected.has(scene.id)) &&
      (!query || scene.id.includes(query))
    );
    gallery.replaceChildren(...visibleScenes.map(cardFor));
    $("#empty").hidden = !!visibleScenes.length;
    updateSelection();
  }

  function openPreview(id) {
    const scene = sceneMap.get(id);
    if (!scene) return;
    currentScene = id;
    $("#preview-title").textContent = id;
    $("#preview-reference").src = scene.reference;
    $("#preview-raw").src = scene.raw;
    $("#preview-slider").value = 0;
    updateSlider();
    updateSelection();
    if (!previewDialog.open) previewDialog.showModal();
  }

  function nextScene(direction) {
    const ids = visibleScenes.map((scene) => scene.id);
    if (!ids.length) return;
    const index = ids.indexOf(currentScene);
    openPreview(ids[(index + direction + ids.length) % ids.length]);
  }

  function selectionText() {
    return `LC-ISP 网站选图\n右上角：${hero}\n展示场景：${[...selected].join("、")}\n\n${location.href}`;
  }

  async function copySelection() {
    const field = $("#selection-text");
    try {
      await navigator.clipboard.writeText(field.value);
      $("#copy-status").textContent = "已复制，粘贴到聊天里即可。";
    } catch {
      field.focus();
      field.select();
      let copied = false;
      try { copied = document.execCommand("copy"); } catch {}
      $("#copy-status").textContent = copied ? "已复制，粘贴到聊天里即可。" : "请手动复制上面的选择。";
    }
  }

  for (const button of document.querySelectorAll("[data-filter]")) {
    button.addEventListener("click", () => {
      filter = button.dataset.filter;
      for (const item of document.querySelectorAll("[data-filter]")) item.setAttribute("aria-pressed", String(item === button));
      render();
    });
  }
  $("#search").addEventListener("input", render);
  $("#clear").addEventListener("click", () => {
    selected.clear();
    hero = null;
    save();
    render();
  });
  $("#export").addEventListener("click", () => {
    $("#selection-text").value = selectionText();
    $("#copy-status").textContent = "";
    exportDialog.showModal();
  });
  $("#copy").addEventListener("click", copySelection);
  $("#close-export").addEventListener("click", () => exportDialog.close());
  $("#close-preview").addEventListener("click", () => previewDialog.close());
  $("#preview-select").addEventListener("click", () => toggle(currentScene, !selected.has(currentScene)));
  $("#preview-hero").addEventListener("click", () => setHero(currentScene));
  $("#preview-slider").addEventListener("input", updateSlider);
  $("#previous").addEventListener("click", () => nextScene(-1));
  $("#next").addEventListener("click", () => nextScene(1));
  previewDialog.addEventListener("keydown", (event) => {
    if (event.target.matches('input[type="range"]')) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      nextScene(event.key === "ArrowLeft" ? -1 : 1);
    }
  });
  for (const dialog of [previewDialog, exportDialog]) {
    dialog.addEventListener("click", (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
  }
  window.addEventListener("hashchange", () => {
    readHash();
    updateSelection();
    if (filter === "selected") render();
  });
  $("#all-count").textContent = scenes.length;
  $("#loading").hidden = true;
  render();
})();
