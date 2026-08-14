#!/usr/bin/env python3
"""Static dev server for Sotto with correct MIME types and no caching."""
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4173
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.mjs': 'text/javascript',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.webmanifest': 'application/manifest+json',
        '.woff2': 'font/woff2',
        '.svg': 'image/svg+xml',
        '.task': 'application/octet-stream',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))


if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Sotto dev server on http://localhost:{PORT}", flush=True)
        httpd.serve_forever()
