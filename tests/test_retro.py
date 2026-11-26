"""Tests for the retro animation engine (TreeThrowRetro.py) and the GUI module import."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import TreeThrow as tt
import TreeThrowRetro as tr


def small_run(species="deciduous", **kw):
    cfg = dict(length=30, width=12, slope=0.35, dx=0.25, years=400, seed=3)
    cfg.update(kw)
    return tr.run_with_frames(tt.SPECIES[species], tt.HillslopeConfig(**cfg), n_frames=40)


class FontTests(unittest.TestCase):
    def test_glyphs_are_5x7(self):
        for ch, g in tr.FONT.items():
            self.assertEqual(g.shape, (7, 5), ch)
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            self.assertIn(ch, tr.FONT)
            self.assertTrue(tr.FONT[ch].any(), ch)

    def test_draw_text(self):
        img = np.zeros((20, 60), np.uint8)
        end = tr.draw_text(img, 1, 1, "Yr 42", tr.WHITE)
        self.assertEqual(end, 1 + 6 * 5)
        self.assertTrue((img == tr.WHITE).any())


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.frames, cls.meta = small_run()

    def test_frames_cover_run(self):
        self.assertGreaterEqual(len(self.frames), 30)
        self.assertEqual(self.frames[0].t, 0.0)
        self.assertEqual(self.frames[-1].t, self.model.time)
        self.assertTrue(np.all(np.diff([f.t for f in self.frames]) > 0))

    def test_every_uprooting_is_captured_once(self):
        n = sum(len(f.events) for f in self.frames)
        self.assertEqual(n, self.model.n_events)
        ev = np.concatenate([f.events for f in self.frames if len(f.events)])
        np.testing.assert_allclose(ev[:, :9], np.array(self.model.event_log)[:, :9])

    def test_frame_state_matches_model(self):
        f = self.frames[-1]
        self.assertEqual(f.trees["x"].size, int((self.model.age >= 0).sum()))
        self.assertEqual(f.stats["uprootings"], self.model.n_events)
        self.assertAlmostEqual(f.stats["roughness_cm"], 100 * self.model.roughness(), places=6)
        np.testing.assert_allclose(f.prof, self.model.z.mean(axis=1), rtol=1e-6)

    def test_render_uses_ega_palette(self):
        r = tr.RetroRenderer(self.frames, self.meta)
        for i in (0, len(self.frames) // 2, len(self.frames) - 1):
            rgb = r.render(i)
            self.assertEqual(rgb.shape, (tr.H_PX, tr.W_PX, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            cols = {tuple(c) for c in rgb.reshape(-1, 3)}
            self.assertTrue(cols <= {tuple(c) for c in tr.EGA})

    def test_upscale_and_ppm(self):
        rgb = tr.RetroRenderer(self.frames, self.meta).render(3)
        big = tr.upscale(rgb, 3, scanlines=True)
        self.assertEqual(big.shape, (600, 960, 3))
        self.assertTrue((big[2::3] <= big[0::3]).all())
        ppm = tr.to_ppm(rgb)
        self.assertTrue(ppm.startswith(b"P6 320 200 255\n"))
        self.assertEqual(len(ppm), len(b"P6 320 200 255\n") + 320 * 200 * 3)

    def test_save_gif(self):
        r = tr.RetroRenderer(self.frames[:8], self.meta)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.gif"
            tr.save_gif(r, p, scale=1, fps=10)
            from PIL import Image
            im = Image.open(p)
            self.assertEqual(im.size, (320, 200))
            self.assertEqual(getattr(im, "n_frames", 1), 8)


class DisturbanceFrameTests(unittest.TestCase):
    def test_fire_is_always_shown(self):
        # Frame spacing (~50 yr) much coarser than the fire; frames are still captured at the fire.
        model, frames, meta = tr.run_with_frames(
            tt.SPECIES["pine"], tt.HillslopeConfig(length=30, width=12, slope=0.4, years=2000, seed=2,
                                                   disturbance="fire", first_disturbance=503), n_frames=40)
        k = [i for i, f in enumerate(frames) if f.disturbances]
        self.assertEqual(len(k), 1)
        f = frames[k[0]]
        self.assertLessEqual(f.t - 503, 1.0)
        self.assertEqual(f.stats["trees"], 0)
        self.assertGreater(f.stats["snags"], 0)
        later = frames[k[0] + 1:k[0] + 4]
        self.assertTrue(all(fr.t - f.t < 15 for fr in later))     # burst of frames after the fire
        r = tr.RetroRenderer(frames, meta)
        rgb = r.render(k[0])
        reds = (rgb == tr.EGA[tr.RED]).all(axis=2).sum()
        self.assertGreater(reds, 2000)                             # red sky

    def test_outbreak_is_tagged_and_leaves_snags(self):
        model, frames, meta = tr.run_with_frames(
            tt.SPECIES["eastern_hemlock"], tt.HillslopeConfig(length=30, width=12, slope=0.35, years=300, seed=4,
                                                              disturbance="outbreak", first_disturbance=150,
                                                              disturbance_severity=0.8), n_frames=30)
        k = [i for i, f in enumerate(frames) if f.disturbances]
        self.assertEqual(len(k), 1)
        self.assertGreater(frames[k[0]].stats["snags"], 0)
        r = tr.RetroRenderer(frames, meta)
        self.assertEqual(r.outbreak_times, [frames[k[0]].disturbances[0][0]])
        title = r.render(k[0] if k[0] % 2 == 0 else k[0] + 1)[:10]
        self.assertTrue((title == tr.EGA[tr.LMAGENTA]).all(axis=2).any())     # OUTBREAK! banner

    def test_stop_callback(self):
        import threading
        ev = threading.Event()

        def prog(frac, msg):
            if frac > 0.3:
                ev.set()
        model, frames, meta = tr.run_with_frames(tt.SPECIES["aspen"], tt.HillslopeConfig(
            length=20, width=10, years=1000, seed=1), n_frames=50, progress=prog, stop=ev)
        self.assertLess(model.time, 600)
        self.assertEqual(frames[-1].t, model.time)


class SpriteTests(unittest.TestCase):
    def test_every_preset_has_its_own_sprite(self):
        kinds = {k: tr.species_kind(sp) for k, sp in tt.SPECIES.items()}
        self.assertEqual(len(set(kinds.values())), len(tt.SPECIES))
        for kind in kinds.values():
            self.assertIn(kind, tr.HMAX)
            spr = tr.make_tree_sprite(kind, tr.HMAX[kind])
            self.assertEqual(spr.shape[0], tr.HMAX[kind])
            self.assertTrue((spr != tr.CLEAR).any())
        self.assertGreater(tr.make_tree_sprite("oak", 20).shape[1], tr.make_tree_sprite("deciduous", 20).shape[1])
        self.assertLess(tr.make_tree_sprite("fir", 20).shape[1], tr.make_tree_sprite("spruce", 20).shape[1])
        custom = tt.replace(tt.SPECIES["pine"], key="my_hemlock", name="Tsuga stand")
        self.assertEqual(tr.species_kind(custom), "hemlock")


@unittest.skipUnless(importlib.util.find_spec("tkinter") is not None, "tkinter not available")
class GuiImportTests(unittest.TestCase):
    def test_gui_module_imports_and_parameters_are_valid(self):
        import TreeThrowGUI as G
        from dataclasses import fields
        sp_fields = {f.name for f in fields(tt.TreeSpecies)}
        cfg_fields = {f.name for f in fields(tt.HillslopeConfig)}
        for spec in (G.FOREST, G.HILLSLOPE, G.PROCESSES, G.DISTURBANCE):
            for _, items in spec:
                for target, key, *_ in items:
                    self.assertIn(key, sp_fields if target == "sp" else cfg_fields)


if __name__ == "__main__":
    unittest.main()
