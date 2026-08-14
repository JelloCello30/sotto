/* Sotto — landing page behavior.
   Two jobs: render the decorative lip-wave motifs, and run the accessible
   install tabs. No frameworks, no network, no state. */

/* ---------- lip-wave ----------
   A row of thin vertical bars whose heights drift like a mouth-openness
   signal. Heights follow a bounded random walk so neighboring bars are
   correlated (organic, signal-like) rather than pure noise. A handful of
   bars are "calm" — tiny amplitude, long period — so the wave is
   occasionally near-still, the way a mouth is. Reduced-motion users get
   the bars frozen at their mid heights (handled in landing.css). */

function renderLipwave(el) {
  const count = Math.max(8, Number.parseInt(el.dataset.bars || '48', 10) || 48);

  // Pick a few accent bars, spread out rather than clumped.
  const accentCount = Math.max(2, Math.round(count / 12));
  const accents = new Set();
  while (accents.size < accentCount) {
    accents.add(Math.floor(Math.random() * count));
  }

  const frag = document.createDocumentFragment();
  let mid = 0.3 + Math.random() * 0.25;

  for (let i = 0; i < count; i++) {
    // Random walk keeps adjacent bars related.
    mid += (Math.random() - 0.5) * 0.16;
    mid = Math.min(0.72, Math.max(0.14, mid));

    const calm = Math.random() < 0.16;
    const amp = calm ? 0.03 + Math.random() * 0.04 : 0.18 + Math.random() * 0.5;
    const peak = Math.min(1, mid + amp);
    const low = Math.max(0.05, mid - amp * (0.5 + Math.random() * 0.4));
    const mid2 = low + (peak - low) * (0.35 + Math.random() * 0.3);
    const dur = (calm ? 4.5 : 2.4) + Math.random() * 3.4;
    const delay = -(Math.random() * dur); // negative: already mid-cycle on load

    const bar = document.createElement('span');
    bar.className = accents.has(i) ? 'bar is-accent' : 'bar';
    bar.style.setProperty('--mid', mid.toFixed(3));
    bar.style.setProperty('--peak', peak.toFixed(3));
    bar.style.setProperty('--low', low.toFixed(3));
    bar.style.setProperty('--mid2', mid2.toFixed(3));
    bar.style.setProperty('--dur', `${dur.toFixed(2)}s`);
    bar.style.setProperty('--delay', `${delay.toFixed(2)}s`);
    frag.appendChild(bar);
  }

  el.replaceChildren(frag);
}

/* ---------- install tabs ----------
   Standard ARIA tabs: roving tabindex, selection follows focus, arrow keys
   plus Home/End. The initially selected tab is chosen from the visitor's
   platform so the relevant instructions are already open. */

function detectPlatformId() {
  const ua = navigator.userAgent;
  // iPadOS Safari reports "Macintosh"; touch points tell it apart.
  if (/iPhone|iPad|iPod/.test(ua)) return 'ios';
  if (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1) return 'ios';
  if (/Android/.test(ua)) return 'android';
  if (/Windows/.test(ua)) return 'windows';
  if (/CrOS|X11|Linux/.test(ua)) return 'linux';
  return 'macos';
}

function initTabs(root) {
  const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
  if (tabs.length === 0) return;
  const panels = tabs.map((tab) =>
    document.getElementById(tab.getAttribute('aria-controls'))
  );

  function select(next, focus = false) {
    tabs.forEach((tab, i) => {
      const on = i === next;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
      if (panels[i]) panels[i].hidden = !on;
    });
    if (focus) tabs[next].focus();
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(i));
    tab.addEventListener('keydown', (event) => {
      const last = tabs.length - 1;
      let next = null;
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          next = i === last ? 0 : i + 1;
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          next = i === 0 ? last : i - 1;
          break;
        case 'Home':
          next = 0;
          break;
        case 'End':
          next = last;
          break;
        default:
          return;
      }
      event.preventDefault();
      select(next, true);
    });
  });

  const platform = detectPlatformId();
  const initial = tabs.findIndex((tab) => tab.dataset.platform === platform);
  select(initial === -1 ? 0 : initial);
}

/* ---------- boot ---------- */

document.querySelectorAll('[data-lipwave]').forEach(renderLipwave);

const tabsRoot = document.querySelector('[data-tabs]');
if (tabsRoot) initTabs(tabsRoot);
