"use strict";

document.querySelectorAll(".comparison").forEach((comparison) => {
  const slider = comparison.querySelector('input[type="range"]');

  function update() {
    const value = Math.max(0, Math.min(100, Number(slider.value)));
    comparison.style.setProperty("--position", `${value}%`);
    comparison.dataset.endpoint = value === 0 ? "reference" : value === 100 ? "raw" : "split";
    slider.setAttribute("aria-valuetext", `${value}% RAW, ${100 - value}% Reference`);
  }

  slider.addEventListener("input", update);
  comparison.querySelectorAll("[data-position]").forEach((button) => {
    button.addEventListener("click", () => {
      slider.value = button.dataset.position;
      update();
    });
  });
  update();
});
