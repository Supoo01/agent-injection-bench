"""Unit-Tests Injektions-Pruefstand: Versionserfassung, Plan, Auswertung, Wache, Kette, Lauf (ohne GPU/Netz).

Aufruf: venv/bin/python -m unittest discover -s pruefstand/tests -t . -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pruefstand import auswertung as aw  # noqa: E402
from pruefstand import plan as pl  # noqa: E402
from pruefstand import version as vs  # noqa: E402

P = ROOT / "pruefstand"


def fake_snapshot(commit):
    return {"commit": commit.ljust(40, "0"), "worktree": "/x",
            "files_sha256": {"agent/session_gateway.py": "aa", "agent/session_gateway_critic.py": "cc"}}


class FakeAdapter:
    def __init__(self, root: Path, rel="data/lora_x", r=8, payload=b"weights"):
        d = root / rel
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(payload)
        (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": 16, "base_model_name_or_path": "b"}))
        self.rel = rel


class VersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.harness = self.tmp / "harness"
        self.models = self.tmp / "models"
        rev = self.models / "hub" / f"models--{vs.BASE_MODEL.replace('/', '--')}" / "refs"
        rev.mkdir(parents=True)
        (rev / "main").write_text("abc123\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def rec(self, spec):
        return vs.erfasse(spec, harness=self.harness, models=self.models, snapshot=fake_snapshot,
                          docker=lambda cmd: "sha256:img")

    def test_presets(self):
        n = vs.normalize_spec({"modell": "basis", "gateway": "r4"})
        self.assertEqual(n["gateway"]["commit"], "d8c3aafb22")
        self.assertEqual(n["gateway"]["schalter"]["verfeinerung"], "r4")
        self.assertIsNone(vs.normalize_spec({"modell": "basis", "gateway": "aus"})["gateway"])
        self.assertEqual(vs.normalize_spec({"modell": "basis", "gateway": "streng"})["gateway"]["commit"], "c8341ea748")

    def test_invalid_specs(self):
        for spec in ({"modell": "basis", "gateway": "r9"}, {"modell": "", "gateway": "aus"},
                     {"modell": "basis", "gateway": {"commit": "x"}},
                     {"modell": "basis", "gateway": "aus", "critic": {"version": "v2"}},
                     {"modell": "basis", "gateway": "r4", "critic": {"version": "v3"}},
                     {"modell": "basis", "gateway": "r4",
                      "critic": {"version": "v2", "optionen": {"base_url": "http://example.com/v1"}}}):
            with self.assertRaises(vs.VersionFehler, msg=spec):
                vs.normalize_spec(spec)

    def test_basis_record(self):
        r = self.rec({"modell": "basis", "gateway": "r4"})
        self.assertEqual(r["modell"]["art"], "basis")
        self.assertEqual(r["modell"]["served_name"], "coding")
        self.assertEqual(r["modell"]["basis_revision"], "abc123")
        self.assertEqual(r["gateway"]["schalter"]["verfeinerung"], "r4")
        self.assertEqual(r["umgebung"]["vllm_image_id"], "sha256:img")
        self.assertEqual(len(r["version_id"]), 16)

    def test_adapter_sha_and_id(self):
        a = FakeAdapter(self.harness)
        r1 = self.rec({"modell": a.rel, "gateway": "r4"})
        self.assertEqual(r1["modell"]["served_name"], "hermes")
        self.assertEqual(r1["modell"]["adapter"]["sha256"]["adapter_model.safetensors"],
                         vs.sha256_file(self.harness / a.rel / "adapter_model.safetensors"))
        self.assertEqual(r1["version_id"], self.rec({"modell": a.rel, "gateway": "r4"})["version_id"])  # stabil
        (self.harness / a.rel / "adapter_model.safetensors").write_bytes(b"other")
        self.assertNotEqual(r1["version_id"], self.rec({"modell": a.rel, "gateway": "r4"})["version_id"])
        # absoluter Pfad im Repo wird angenommen
        self.assertEqual(self.rec({"modell": str(self.harness / a.rel), "gateway": "r4"})["modell"]["adapter"]["pfad"], a.rel)

    def test_id_depends_on_gateway_and_critic(self):
        ids = {self.rec(s)["version_id"] for s in (
            {"modell": "basis", "gateway": "r4"}, {"modell": "basis", "gateway": "aus"},
            {"modell": "basis", "gateway": "streng"},
            {"modell": "basis", "gateway": "r4", "critic": {"version": "v2"}},
            {"modell": "basis", "gateway": "r4", "critic": {"version": "v2", "optionen": {"timeout_s": 45}}})}
        self.assertEqual(len(ids), 5)

    def test_adapter_errors(self):
        with self.assertRaises(vs.VersionFehler):
            self.rec({"modell": "data/fehlt", "gateway": "r4"})
        with self.assertRaises(vs.VersionFehler):
            self.rec({"modell": "../aussen", "gateway": "r4"})
        a = FakeAdapter(self.harness, "data/gross", r=32)
        with self.assertRaises(vs.VersionFehler):
            self.rec({"modell": a.rel, "gateway": "r4"})

    def test_critic_record(self):
        r = self.rec({"modell": "basis", "gateway": "r4", "critic": {"version": "v2"}})
        self.assertEqual(r["critic"]["version"], "v2")
        self.assertEqual(r["critic"]["datei_sha256"], "cc")
        sw = r["gateway"]["schalter"]
        self.assertTrue(sw["alignment_critic"])
        self.assertEqual(sw["alignment_critic_version"], "v2")
        self.assertEqual(sw["critic"]["base_url"], "http://localhost:8012/v1")

    def test_real_snapshot_r4(self):
        """Echter Export aus dem Worktree (git show, nur lesen): Commit voll, Dateien per sha256."""
        r = vs.erfasse({"modell": "basis", "gateway": "r4"}, models=self.models, docker=lambda c: "x")
        self.assertTrue(r["gateway"]["commit"].startswith("d8c3aafb22"))
        self.assertIn("agent/session_gateway_refine.py", r["gateway"]["dateien_sha256"])
        self.assertTrue(all(len(h) == 64 for h in r["gateway"]["dateien_sha256"].values()))

    def test_chat_template_e12(self):
        """E12: optionale Chat-Vorlage -> eigener Eintrag + eigene version_id; ohne Feld bleibt die ID wie bisher."""
        a = FakeAdapter(self.harness)
        ohne = self.rec({"modell": a.rel, "gateway": "aus"})
        self.assertNotIn("chat_template", ohne)
        self.assertNotIn("chat_template", ohne["spec"])
        self.assertNotIn("chat_template_sha256", vs.kern(ohne))
        v = self.harness / "vorlagen" / "e12" / "chat_template.jinja"
        v.parent.mkdir(parents=True)
        v.write_text("{{ messages }}")
        mit = self.rec({"modell": a.rel, "gateway": "aus", "chat_template": "vorlagen/e12/chat_template.jinja"})
        self.assertEqual(mit["chat_template"], {"pfad": "vorlagen/e12/chat_template.jinja", "sha256": vs.sha256_file(v)})
        self.assertNotEqual(mit["version_id"], ohne["version_id"])
        self.assertIn("Vorlage vorlagen/e12/chat_template.jinja", vs.kurzbeschreibung(mit))
        v.write_text("{{ messages }} ")
        self.assertNotEqual(self.rec({"modell": a.rel, "gateway": "aus", "chat_template": "vorlagen/e12/chat_template.jinja"})["version_id"],
                            mit["version_id"])
        for bad in ("/abs/x.jinja", "../x.jinja", "vorlagen/x.txt", 3):
            with self.assertRaises(vs.VersionFehler, msg=bad):
                vs.normalize_spec({"modell": "basis", "chat_template": bad})
        with self.assertRaises(vs.VersionFehler):
            self.rec({"modell": "basis", "chat_template": "vorlagen/fehlt.jinja"})

    def test_version_files_valid(self):
        for f in sorted((P / "versionen").glob("*.json")):
            vs.normalize_spec(json.loads(f.read_text()))


class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = pl.build_plan("schnell")
        cls.v = pl.build_plan("voll")

    def test_schnell_size_and_coverage(self):
        self.assertEqual(len(self.s["items"]), 184)
        self.assertEqual({it["suite"] for it in self.s["items"]}, set(pl.SUITES))
        self.assertEqual({it["attack"] for it in self.s["items"] if it["attack"]}, set(pl.ATTACKS))
        for suite in pl.SUITES:  # jede (nicht ausgeschlossene) Injection-Task kommt vor
            _, injs = pl.suite_tasks(suite)
            got = {p[1] for p in self.s["auswahl"][suite]["paare"]}
            self.assertEqual(got, set(injs), suite)
        keys = [it["key"] for it in self.s["items"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_paired_design(self):
        for suite in pl.SUITES:
            per = {}
            for it in self.s["items"]:
                if it["suite"] == suite and it["attack"]:
                    per.setdefault(it["attack"], set()).add((it["user_task"], it["injection_task"]))
            self.assertEqual(len({frozenset(v) for v in per.values()}), 1)

    def test_deterministic(self):
        self.assertEqual(pl.build_plan("schnell")["plan_sha256"], self.s["plan_sha256"])
        self.assertNotEqual(pl.build_plan("schnell", seed=1)["plan_sha256"], self.s["plan_sha256"])

    def test_workspace_exclusion(self):
        for p in (self.s, self.v):
            bad = [it for it in p["items"] if it["suite"] == "workspace" and it["injection_task"]
                   and int(it["injection_task"].rsplit("_", 1)[1]) >= 6]
            self.assertEqual(bad, [])

    def test_voll(self):
        self.assertEqual(self.v["reps"], 5)
        self.assertEqual(len(self.v["items"]), 5 * (97 + 4 * 629))
        est = pl.estimate(self.v)
        self.assertGreater(est["minuten_je_version"], 40 * 60)

    def test_estimate_schnell_about_one_hour(self):
        est = pl.estimate(self.s)
        self.assertTrue(40 <= est["minuten_je_version"] <= 80, est)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            pl.build_plan("mittel")


def row(suite="slack", inj=None, attack=None, utility=False, security=None, blocked=(), error=None, rep=1, key=None):
    return {"suite": suite, "injection_task": inj, "attack": attack, "utility": utility, "security": security,
            "security_attempted": security, "blocked": list(blocked), "error": error, "rep": rep,
            "key": key or f"{suite}|{inj}|{attack}|{rep}|{id(object())}"}


class AuswertungTest(unittest.TestCase):
    def test_wilson(self):
        self.assertEqual(aw.wilson(0, 10), [0.0, 0.2775])
        self.assertEqual(aw.wilson(5, 10), [0.2366, 0.7634])
        self.assertEqual(aw.wilson(10, 10), [0.7225, 1.0])
        self.assertIsNone(aw.wilson(0, 0))

    def test_kennzahlen(self):
        rows = [row(utility=True), row(utility=False, blocked=["x"]),
                row(inj="i1", attack="a", security=True), row(inj="i1", attack="a", security=False, utility=True),
                row(inj="i2", attack="b", security=False, error="LLM kaputt", suite="banking"),
                row(inj="i2", attack="b", security=False, blocked=["y", "z"], suite="banking")]
        k = aw.kennzahlen(rows)
        self.assertEqual((k["asr"]["k"], k["asr"]["n"]), (1, 4))
        self.assertEqual((k["utility_ohne"]["k"], k["utility_ohne"]["n"]), (1, 2))
        self.assertEqual((k["utility_angriff"]["k"], k["utility_angriff"]["n"]), (1, 4))
        self.assertEqual((k["sperren_ohne"]["k"], k["sperren_angriff"]["k"]), (1, 1))
        self.assertEqual((k["fehler"]["k"], k["fehler"]["n"]), (1, 6))
        self.assertEqual(k["gesperrte_aufrufe"], 3)
        self.assertEqual(k["je_angriff"]["a"]["asr"]["wert"], 0.5)
        self.assertEqual(k["je_suite"]["banking"]["asr"]["wert"], 0.0)
        self.assertNotIn("je_wiederholung", k)
        k2 = aw.kennzahlen(rows + [row(inj="i1", attack="a", security=True, rep=2)])
        self.assertEqual(k2["je_wiederholung"]["asr"], {"min": 0.25, "max": 1.0})

    def test_ablage(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "lauf"
            d.mkdir()
            rows = [row(utility=True, key="k1"), row(inj="i", attack="a", security=True, key="k2"),
                    row(inj="i", attack="a", security=False, key="k2")]  # Duplikat: erster zaehlt
            (d / "ergebnisse.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            version = {"version_id": "abcd", "name": "t", "modell": {"art": "basis"}, "gateway": None, "critic": None,
                       "umgebung": {}}
            plan = {"profil": "schnell", "plan_sha256": "f" * 64}
            e = aw.eintrag(d, version, {"vollstaendig": True, "sekunden": 120}, plan)
            self.assertEqual(e["kennzahlen"]["laeufe"], 2)
            self.assertEqual(e["kennzahlen"]["asr"]["wert"], 1.0)
            v, u, l = tmp / "verlauf.jsonl", tmp / "uebersicht.md", tmp / "status" / "pruefstand_letzter.json"
            aw.ablegen(e, v, u, l)
            aw.ablegen(dict(e, name="t2"), v, u, l)
            self.assertEqual(len(aw.lese_verlauf(v)), 2)
            md = u.read_text()
            self.assertIn("| Zeit | Version |", md)
            self.assertLess(md.index("t2:"), md.index("t:"))  # neueste zuerst
            z = json.loads(l.read_text())
            self.assertEqual(z["name"], "t2")
            self.assertIn("Angriffe 100.0 %", z["zeile"])
            self.assertIn("Angriff erfolgreich", aw.lauf_markdown(e))
        finally:
            shutil.rmtree(tmp)

    def test_default_paths(self):
        self.assertEqual(aw.LETZTER, aw.STATUS_DIR / "pruefstand_letzter.json")
        self.assertEqual(aw.VERLAUF, P / "verlauf.jsonl")


class FakeBin:
    """docker / systemctl / ss als Skripte im PATH, gesteuert ueber Umgebungsvariablen."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp())
        self._w("docker", 'if [ "${FAKE_DOCKER_FAIL:-0}" = 1 ]; then exit 1; fi\nprintf "%b" "${FAKE_CONTAINERS:-}"')
        self._w("systemctl", '''case "$*" in
  *list-units*) printf "%b" "${FAKE_UNITS:-}" ;;
  *is-active*prod-vllm*) echo "${FAKE_PROD:-inactive}" ;;
  *is-active*) echo inactive ;;
esac''')
        self._w("ss", 'printf "%b" "${FAKE_SS:-State Recv-Q Send-Q Local Address:Port\\n}"')

    def _w(self, name, body):
        p = self.dir / name
        p.write_text("#!/usr/bin/bash\n" + body + "\n")
        p.chmod(0o755)

    def env(self, **kw):
        e = {"PATH": f"{self.dir}:/usr/bin:/bin", "HOME": str(self.dir)}
        e.update(kw)
        return e

    def close(self):
        shutil.rmtree(self.dir)


def unitline(name, state="active"):
    return f"{name} loaded {state} running x\\n"


class WacheTest(unittest.TestCase):
    def setUp(self):
        self.fb = FakeBin()

    def tearDown(self):
        self.fb.close()

    def wache(self, **kw):
        r = subprocess.run([str(P / "wache.sh")], env=self.fb.env(**kw), capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout

    def test_frei(self):
        rc, out = self.wache(FAKE_UNITS=unitline("dashboard.service") + unitline("dbus.service"))
        self.assertEqual((rc, out), (0, ""))

    def test_container(self):
        rc, out = self.wache(FAKE_CONTAINERS="searxng\\ninjection-bench-vllm\\n")
        self.assertEqual(rc, 2)
        self.assertIn("container=injection-bench-vllm", out)
        rc, out = self.wache(FAKE_CONTAINERS="mein-vllm-1\\n", PRUEFSTAND_CONTAINER_PREFIX="mein-vllm")
        self.assertIn("container=mein-vllm-1", out)

    def test_units_nur_mit_muster(self):
        units = unitline("train-lora.service") + unitline("eval-gpu.service")
        self.assertEqual(self.wache(FAKE_UNITS=units)[0], 0)  # ohne PRUEFSTAND_BUSY_UNITS keine Unit-Pruefung
        rc, out = self.wache(FAKE_UNITS=units, PRUEFSTAND_BUSY_UNITS="^(train|eval)-")
        self.assertEqual(rc, 2)
        self.assertIn("unit=train-lora.service", out)
        self.assertIn("unit=eval-gpu.service", out)

    def test_eigene_ausgenommen(self):
        units = unitline("injection-bench.service")
        rc, _ = self.wache(FAKE_UNITS=units, PRUEFSTAND_BUSY_UNITS="^injection-bench")
        self.assertEqual(rc, 2)
        rc, out = self.wache(FAKE_UNITS=units, PRUEFSTAND_BUSY_UNITS="^injection-bench", PRUEFSTAND_EIGENE="injection-bench")
        self.assertEqual((rc, out), (0, ""))

    def test_produktion_port_datei_fehler(self):
        self.assertEqual(self.wache(FAKE_PROD="active")[0], 0)  # ohne PRUEFSTAND_PROD_UNIT nicht geprueft
        self.assertIn("produktion=prod-vllm.service:active",
                      self.wache(FAKE_PROD="active", PRUEFSTAND_PROD_UNIT="prod-vllm.service")[1])
        self.assertIn("port8012", self.wache(FAKE_SS="LISTEN 0 4096 0.0.0.0:8012 0.0.0.0:*\\n")[1])
        self.assertEqual(self.wache(FAKE_SS="LISTEN 0 4096 0.0.0.0:18012 0.0.0.0:*\\n")[0], 0)
        self.assertIn("docker-ps-fehlgeschlagen", self.wache(FAKE_DOCKER_FAIL="1")[1])
        self.assertIn("datei_fehlt=", self.wache(PRUEFSTAND_REQUIRE_FILE=str(self.fb.dir / "fehlt"))[1])


class StartSkriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        ad = self.tmp / "adapters" / "my_adapter"
        ad.mkdir(parents=True)
        (ad / "adapter_config.json").write_text("{}")
        (ad / "adapter_model.safetensors").write_text("x")
        (ad / "chat_template.jinja").write_text("{{ x }}")
        self.gen = self.tmp / "gen.json"
        self.gen.write_text('{"temperature": 1.0}')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_dry(self, modell, **kw):
        env = dict(os.environ, PRUEFSTAND_DRY_RUN="1", PRUEFSTAND_HARNESS=str(self.tmp / "adapters"),
                   PRUEFSTAND_GEN_CONFIG=str(self.gen), **kw)
        return subprocess.run([str(P / "start_vllm.sh"), modell], env=env, capture_output=True, text=True, timeout=60)

    def test_lora_flags(self):
        r = self.run_dry("my_adapter")
        self.assertEqual(r.returncode, 0, r.stderr)
        for s in ("--enable-lora", "--lora-modules hermes=/adapters/hermes", "my_adapter:/adapters/hermes:ro",
                  "--name injection-bench-vllm", "--port 8012", "--override-generation-config"):
            self.assertIn(s, r.stdout)
        b = self.run_dry("basis")
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertNotIn("--lora-modules", b.stdout)
        self.assertIn("--enable-lora", b.stdout)  # gleiche Flags fuer alle Versionen

    def test_chat_template_flag(self):
        r = self.run_dry("my_adapter", PRUEFSTAND_CHAT_TEMPLATE="my_adapter/chat_template.jinja")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--chat-template /vorlagen/chat_template.jinja", r.stdout)
        self.assertIn("chat_template.jinja:/vorlagen/chat_template.jinja:ro", r.stdout)
        self.assertNotIn("--chat-template", self.run_dry("my_adapter").stdout)
        for bad in ("../x.jinja", "/etc/passwd", "vorlagen/fehlt.jinja"):
            self.assertNotEqual(self.run_dry("basis", PRUEFSTAND_CHAT_TEMPLATE=bad).returncode, 0, bad)

    def test_rejects(self):
        self.assertNotEqual(self.run_dry("../etc").returncode, 0)
        self.assertNotEqual(self.run_dry("gibt_es_nicht").returncode, 0)
        self.assertNotEqual(self.run_dry("basis", PRUEFSTAND_CONTAINER="fremder-container").returncode, 0)


def probe(check):
    tmp = tempfile.mkdtemp()
    try:
        r = subprocess.run([sys.executable, str(P / "tests" / "_probe.py"), check, tmp], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=900,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
        if r.returncode != 0:
            raise AssertionError(r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(tmp)


class LaufTest(unittest.TestCase):
    """Ende-zu-Ende ohne LLM: Skript-Agent, der Injektionen immer befolgt (Obergrenze)."""

    def test_gateway_aus(self):
        r = probe("aus")
        self.assertIsNone(r["gateway"])
        self.assertTrue(r["meta"]["vollstaendig"])
        att = [x for x in r["rows"] if x["attack"]]
        self.assertTrue(all(x["security"] for x in att))  # ohne Schutz kommt der gehorsame Agent durch
        self.assertEqual(r["neu_bei_fortsetzung"], 0)

    def test_gateway_r4(self):
        r = probe("r4")
        self.assertTrue(r["gateway"].startswith("d8c3aafb22"))
        self.assertEqual(len(r["rows"]), 4)
        self.assertEqual({x["attack"] for x in r["rows"]}, {None, "important_instructions", "tool_knowledge", "ignore_previous"})
        att = [x for x in r["rows"] if x["attack"]]
        self.assertFalse(any(x["security"] for x in att))
        self.assertTrue(all(x["blocked"] for x in att))
        self.assertFalse(any(x["error"] for x in r["rows"]))
        self.assertEqual(r["neu_bei_fortsetzung"], 0)

    def test_critic_v2(self):
        r = probe("critic_v2")
        self.assertTrue(r["gateway"].startswith("7df656f4a5"))
        self.assertTrue(r["critic_calls"])
        self.assertTrue(all(c["v2"] and c["base_url"] == "http://localhost:8012/v1" for c in r["critic_calls"]))
        self.assertTrue(all(x["critic"]["vetoes"] >= 1 for x in r["rows"]))
        self.assertTrue(any(x["critic"]["halt"] for x in r["rows"]))  # 3. Veto beendet den Zug


if __name__ == "__main__":
    unittest.main()
