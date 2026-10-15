"""Bruecke v2 (14.10.2026): URL-Lesetools bekommen echte Hermes-Ziele; Versionsfeld bruecke."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def probe(art):
    r = subprocess.run([sys.executable, str(ROOT / "pruefstand/tests/_probe_bruecke.py"), art], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class BrueckeGateTest(unittest.TestCase):
    def test_ohne_bruecke_buendelt_werkzeug(self):
        d = probe("ohne")   # dokumentiert den K5-Fall (alte Bruecke)
        self.assertEqual([a["freigabe"] for a in d["aufrufe"][:2]], ["granted", None])
        self.assertIn("kanal:tool:agentdojo.get_webpage", d["liste"])

    def test_mit_bruecke_fragt_je_ziel(self):
        d = probe("mit")
        self.assertEqual(d["tools"], ["agentdojo.get_webpage"])
        a = d["aufrufe"]
        self.assertEqual(a[0]["freigabe"], "granted")
        self.assertEqual(a[1]["freigabe"], "granted")      # neues Ziel -> erneut gefragt
        self.assertTrue(a[2]["frei"])                      # bereits freigegebenes Ziel -> gebuendelt
        self.assertFalse(any(e.startswith("kanal:tool:") for e in d["liste"]))
        self.assertTrue(any("eve-blog" in e for e in d["liste"]))


class BrueckeVersionTest(unittest.TestCase):
    def erf(self, spec):
        from pruefstand.version import erfasse

        return erfasse(spec, docker=lambda *a, **k: "img")

    def test_id_nur_mit_feld_anders(self):
        base = {"name": "x", "modell": "basis", "gateway": "s2s5", "mensch": {"p": 0.0}}
        a = self.erf(dict(base))
        b = self.erf(dict(base, bruecke={"url_ziele": True}))
        c = self.erf(dict(base, bruecke={"url_ziele": False}))
        self.assertNotEqual(a["version_id"], b["version_id"])
        self.assertEqual(a["version_id"], c["version_id"])
        from pruefstand.version import kurzbeschreibung

        self.assertIn("Bruecke url_ziele", kurzbeschreibung(b))

    def test_ohne_feld_deterministisch(self):
        import json as _j

        spec = _j.loads((ROOT / "pruefstand/versionen/basis_s2s5_mensch0.json").read_text())
        self.assertEqual(self.erf(dict(spec))["version_id"], self.erf(dict(spec))["version_id"])
        self.assertNotIn("bruecke", self.erf(dict(spec)))

    def test_ungueltig(self):
        from pruefstand.version import VersionFehler

        with self.assertRaises(VersionFehler):
            self.erf({"name": "x", "modell": "basis", "gateway": "s2s5", "bruecke": {"anders": True}})


if __name__ == "__main__":
    unittest.main()
