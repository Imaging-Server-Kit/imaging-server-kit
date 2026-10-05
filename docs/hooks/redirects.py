"""MkDocs hook writing redirect pages for the URLs of the former Jupyter Book site.

The mapping (old URL -> new page) is read from `extra.redirects` in mkdocs.yml.
"""

from pathlib import Path
import posixpath

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting...</title>
<link rel="canonical" href="{url}">
<meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
<p>This page has moved to <a href="{url}">{url}</a>.</p>
</body>
</html>
"""


def on_post_build(config, **kwargs):
    redirects = config["extra"].get("redirects", {})
    site_dir = Path(config["site_dir"])
    for old_url, new_page in redirects.items():
        new_url = str(Path(new_page).with_suffix(".html").as_posix())
        relative_url = posixpath.relpath(new_url, posixpath.dirname(old_url))
        old_file = site_dir / old_url
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text(TEMPLATE.format(url=relative_url), encoding="utf-8")
