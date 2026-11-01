// Ciel's service worker makes the browser app installable and shows a friendly screen
// when the PC can't be reached. It deliberately stores nothing: no Cache Storage, and it
// never touches API requests, so chats and settings only ever come live from the PC.
'use strict';

const OFFLINE_PAGE = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1C1B20"><title>Ciel</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; min-height: 100vh; min-height: 100dvh; display: grid; place-items: center;
    background: #1C1B20; color: #ECEAF2; font: 16px/1.5 system-ui, sans-serif;
    padding: env(safe-area-inset-top) 24px env(safe-area-inset-bottom); box-sizing: border-box; }
  main { max-width: 22rem; text-align: center; }
  .orb { width: 88px; height: 88px; margin: 0 auto 28px; border-radius: 50%; opacity: .5;
    background: radial-gradient(circle at 34% 30%, #DAEDFF, #A480FA 38%, #8B5CF6 62%, #67E8F9);
    box-shadow: 0 0 48px rgba(139, 92, 246, .35); }
  h1 { font-size: 1.25rem; margin: 0 0 8px; }
  p { margin: 0 0 24px; color: #A9A6B4; }
  a { display: inline-block; padding: 12px 28px; border-radius: 999px; background: #ECEAF2;
    color: #1C1B20; font-weight: 600; text-decoration: none; }
</style></head>
<body><main>
  <div class="orb" aria-hidden="true"></div>
  <h1>Ciel's PC isn't reachable</h1>
  <p>Check that the PC is on, Ciel is running, and this phone is on the same Wi-Fi.</p>
  <a href="/">Try again</a>
</main></body></html>`;

self.addEventListener('install', () => { self.skipWaiting(); });
self.addEventListener('activate', event => {
  if (event && event.waitUntil) event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.mode !== 'navigate' || request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;
  event.respondWith(fetch(request).catch(() => new Response(OFFLINE_PAGE, {
    status: 503,
    headers: {'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store'},
  })));
});
