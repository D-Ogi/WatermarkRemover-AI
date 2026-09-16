"""Keep the desktop shell self-contained and verify its vendored asset inventory."""
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]


class Assets(HTMLParser):
    """Collect executable scripts and stylesheets from the real page."""
    def __init__(self):
        """Start an empty inventory before parsing the application HTML."""
        super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        """Record only resources loaded automatically by the browser."""
        attrs=dict(attrs)
        if tag=='script' and 'src' in attrs:self.paths.append(attrs['src'])
        if tag=='link' and attrs.get('rel')=='stylesheet':self.paths.append(attrs['href'])


def test_desktop_scripts_and_styles_are_local():
    """A first launch must not depend on CDNs for its script or stylesheet assets."""
    parser=Assets();parser.feed((ROOT/'ui/index.html').read_text(encoding='utf-8'))
    assert parser.paths
    for name in parser.paths:
        assert not urlparse(name).scheme and not name.startswith('//')
        assert (ROOT/'ui'/name).is_file()


def test_vendored_assets_match_manifest():
    """Catch missing or accidentally changed vendor files, including fonts/licenses."""
    vendor=ROOT/'ui/vendor';manifest=json.loads((vendor/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==digest,name


def test_ui_and_package_version_agree():
    """Use one release version for the launcher and displayed interface."""
    assert (ROOT/'VERSION').read_text().strip()==json.loads((ROOT/'ui/config.json').read_text(encoding='utf-8'))['version']


def test_theme_and_font_resources_are_local():
    """Every theme and font must remain usable without an external image/font server."""
    for stylesheet in [ROOT/'ui/themes.css', ROOT/'ui/vendor/fonts.css']:
        for value in re.findall(r"url\(([^)]+)\)", stylesheet.read_text(encoding='utf-8')):
            value=value.strip("\"'")
            assert not urlparse(value).scheme and not value.startswith('//'),value
            assert (stylesheet.parent/value).is_file(),value
