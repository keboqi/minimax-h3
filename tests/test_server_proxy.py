"""Focused CPU contracts retained from the former launcher self-test."""

import unittest
import unittest.mock

import fastapi as _fastapi
import httpx as _httpx

import h3_app.server as _server


class ServerProxyTests(unittest.TestCase):
    def test_proxy_paths_headers_and_cookies(self):
        rewritten_html = _server._rewrite_comfy_text(
            (
                '<html><head></head><body><script src="/assets/app.js">'
                "</script></body></html>"
            ),
            "text/html; charset=utf-8",
        )
        self.assertIn('<base href="/comfyui/">', rewritten_html)
        self.assertIn('src="/comfyui/assets/app.js"', rewritten_html)
        rewritten_css = _server._rewrite_comfy_text(
            'src:url("/assets/font.woff2")',
            "text/css",
        )
        self.assertIn('url("/comfyui/assets/font.woff2")', rewritten_css)
        self.assertEqual(
            _server._rewrite_comfy_text(
                "const route = '/api';", "application/javascript"
            ),
            "const route = '/api';",
        )
        self.assertEqual(
            _server._comfy_upstream_path(
                "api/userdata/workflows/LTX 2.5/example.json",
                b"/comfyui/api/userdata/workflows%2FLTX%202.5%2Fexample.json",
            ),
            "api/userdata/workflows%2FLTX%202.5%2Fexample.json",
        )
        self.assertEqual(
            _server._comfy_upstream_path("assets/app.js", b"/comfyui/assets/app.js"),
            "assets/app.js",
        )
        existing_base = _server._rewrite_comfy_text(
            '<html><head><base href="/custom/"></head></html>',
            "text/html",
        )
        self.assertEqual(existing_base.count("<base "), 1)
        filtered_headers = _server._proxy_headers(
            {
                "Connection": "keep-alive, x-private",
                "X-Private": "drop",
                "X-Test": "ok",
            }
        )
        self.assertEqual(filtered_headers, {"X-Test": "ok"})
        cookie_response = _server._append_set_cookies(
            _fastapi.Response(),
            _httpx.Headers(
                [
                    ("set-cookie", "one=1; Path=/"),
                    ("set-cookie", "two=2; Path=/"),
                ]
            ),
        )
        self.assertEqual(
            cookie_response.headers.getlist("set-cookie"),
            ["one=1; Path=/", "two=2; Path=/"],
        )
