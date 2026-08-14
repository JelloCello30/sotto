/* Sotto — service worker.
 *
 * Strategy:
 *   - Precache the full app shell plus all vendored assets at install.
 *     Each file is cached individually so a single miss cannot fail the
 *     whole install; misses are logged and retried lazily by the fetch
 *     handler later.
 *   - vendor/* and assets/* are immutable in practice: cache-first.
 *   - Everything else (HTML, CSS, JS, manifest): network-first, falling
 *     back to cache so the app keeps working offline.
 *   - Old sotto-* caches are deleted on activate.
 */

const CACHE = 'sotto-v1';

const PRECACHE = [
  './',
  'index.html',
  'app.html',
  'css/styles.css',
  'css/landing.css',
  'css/app.css',
  'js/engine.js',
  'js/app.js',
  'js/landing.js',
  'manifest.webmanifest',
  'assets/logo.svg',
  'assets/icon.svg',
  'assets/icon-192.png',
  'assets/icon-512.png',
  'assets/icon-maskable-192.png',
  'assets/icon-maskable-512.png',
  'vendor/inter/inter-var.woff2',
  'vendor/mediapipe/vision_bundle.js',
  'vendor/mediapipe/face_landmarker.task',
  'vendor/mediapipe/wasm/vision_wasm_internal.js',
  'vendor/mediapipe/wasm/vision_wasm_internal.wasm',
  'vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js',
  'vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) =>
        Promise.all(
          PRECACHE.map((url) =>
            cache.add(url).catch((err) => {
              // Do not fail the whole install over one file; log and move on.
              console.warn('[sotto-sw] precache miss:', url, err.message || err);
            })
          )
        )
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('sotto-') && key !== CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

const CACHE_FIRST = /\/(?:vendor|assets)\//;

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // Sotto makes none; do not intercept.

  if (CACHE_FIRST.test(url.pathname)) {
    event.respondWith(cacheFirst(req));
  } else {
    event.respondWith(networkFirst(req));
  }
});

async function cacheFirst(req) {
  const cached = await caches.match(req, { ignoreSearch: true });
  if (cached) return cached;
  const res = await fetch(req);
  if (res.ok) {
    const cache = await caches.open(CACHE);
    cache.put(req, res.clone());
  }
  return res;
}

async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (err) {
    const cached = await cache.match(req, { ignoreSearch: true });
    if (cached) return cached;
    if (req.mode === 'navigate') {
      // Last resort for offline navigations to uncached URLs: the shell.
      const shell = await cache.match('./');
      if (shell) return shell;
    }
    throw err;
  }
}
