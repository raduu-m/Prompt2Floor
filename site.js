document.querySelectorAll('video').forEach((video) => {
  video.muted = true;
  const playback = video.play();
  if (playback) playback.catch(() => {});
});

document.querySelectorAll('[data-comparison]').forEach((comparison) => {
  const range = comparison.querySelector('input[type="range"]');
  const left = comparison.querySelector('[data-side="left"]');
  const right = comparison.querySelector('[data-side="right"]');
  const view = comparison.querySelector('.comparison-view');
  const examples = [...comparison.querySelectorAll('[data-example]')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let activeExample = examples[0].dataset.example;
  let animation = 0;
  let visible = false;
  let introduced = false;
  const names = {pretrained: 'Pretrained', id: 'ID', ood: 'OOD', mixed: 'Mixed (ID + OOD)'};
  function updateSide(side, select) {
    comparison.querySelector(`[data-image="${side}"]`).src = `assets/comparison/${activeExample}/${select.value}.jpg`;
    comparison.querySelector(`[data-image="${side}"]`).alt = `${names[select.value]} person detections on the same held-out warehouse frame`;
    comparison.querySelector(`[data-label="${side}"]`).textContent = names[select.value];
  }
  function updatePosition() {
    comparison.style.setProperty('--split', `${range.value}%`);
    range.setAttribute('aria-valuetext', `${range.value}% left image visible`);
  }
  function stopHint() { cancelAnimationFrame(animation); animation = 0; }
  function hint() {
    stopHint();
    if (reducedMotion.matches || !visible) return;
    const center = Number(range.value);
    let start;
    function tick(time) {
      if (start === undefined) start = time;
      const progress = Math.min((time - start) / 1900, 1);
      // One gentle sweep in each direction, easing back to the chosen position.
      range.value = String(center + 9 * Math.sin(progress * 2 * Math.PI) * Math.sin(progress * Math.PI));
      updatePosition();
      if (progress < 1) animation = requestAnimationFrame(tick);
      else { range.value = String(center); updatePosition(); animation = 0; }
    }
    animation = requestAnimationFrame(tick);
  }
  examples.forEach((button) => {
    button.addEventListener('click', () => {
      stopHint(); activeExample = button.dataset.example;
      examples.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      left.value = button.dataset.left; right.value = button.dataset.right;
      range.value = button.dataset.split;
      updateSide('left', left); updateSide('right', right); updatePosition(); hint();
    });
  });
  left.addEventListener('change', () => { updateSide('left', left); hint(); });
  right.addEventListener('change', () => { updateSide('right', right); hint(); });
  ['pointerdown', 'keydown', 'focus'].forEach((event) => range.addEventListener(event, stopHint));
  range.addEventListener('input', () => { stopHint(); updatePosition(); });
  reducedMotion.addEventListener('change', stopHint);
  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    if (visible && !introduced) { introduced = true; hint(); }
    if (!visible) stopHint();
  }, {threshold: .4}).observe(view);
  updateSide('left', left); updateSide('right', right); updatePosition();
});
