#!/usr/bin/env python3
"""
TreeThrowGUI.py — graphical front end for the tree-throw hillslope model
========================================================================
    python TreeThrowGUI.py

Left: model parameters, grouped in tabs
    Forest       species preset, stand dynamics, uprooting, root plate and pit
                 (with live derived rates for the chosen settings)
    Hillslope    geometry, run length, soil and bedrock
    Processes    creep, tree-throw mechanics, pit-mound collapse, root cycling
    Disturbance  fire / windstorm regime
    Output       animation frames, pixel scale, results folder
Right: results
    Retro view   1980s pixel animation of the run (play, scrub, export GIF/PNG)
    Results      the eight-panel scientific figure
    Summary      key numbers from the run
    Log          messages

The run happens in a background thread; the window stays responsive and the
run can be stopped.  Requires tkinter, numpy and matplotlib (Pillow, which
ships with matplotlib, is used for GIF/PNG export).
"""

from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

import TreeThrow as tt  # noqa: E402
import TreeThrowRetro as tr  # noqa: E402

APP_TITLE = "Tree-Throw Hillslope Model"
CUSTOM = "custom (from JSON)"

# ----------------------------------------------------------------------------
# Parameter definitions: (target, key, label, kind, tooltip)
#   target: "sp" = TreeSpecies field, "cfg" = HillslopeConfig field, "anim" = GUI option
#   kind:   "float", "int", "bool", "optfloat" (blank = None) or a tuple of choices
# ----------------------------------------------------------------------------
FOREST = [
    ("Stand dynamics", [
        ("sp", "density", "Stand density (stems/ha)", "float", "Number of trees per hectare, held constant."),
        ("sp", "lifespan_mean", "Mean lifespan (yr)", "float", "Senescence age; trees die at this age if nothing kills them first."),
        ("sp", "lifespan_sd", "Lifespan s.d. (yr)", "float", "Spread of senescence ages."),
        ("sp", "background_mortality", "Background mortality (/yr)", "float",
         "Annual probability of death at any age (competition, disease). Šamonil et al. 2017: ~0.7-2 %/yr."),
        ("sp", "regeneration_delay", "Regeneration delay (yr)", "float", "Mean gap time before a dead tree is replaced."),
        ("sp", "maturity_age", "Maturity age (yr)", "float", "Age at ~95 % of mature root-plate diameter."),
        ("cfg", "stand_structure", "Initial stand", ("all-aged", "even-aged"),
         "all-aged = steady state; even-aged = a young cohort, e.g. after a disturbance."),
    ]),
    ("Uprooting", [
        ("sp", "uproot_fraction", "P(uproot | eligible death)", "float",
         "Probability that a dying tree old enough to uproot is uprooted (otherwise it snaps or dies standing)."),
        ("sp", "min_uproot_age", "Minimum uprooting age (yr)", "float", "Younger trees snap instead (Gabet & Mudd 2010: 80 yr)."),
        ("sp", "fall_slope_bias", "Downslope fall bias", "float",
         "von Mises concentration per unit gradient; 2 reproduces Gabet & Mudd's 50 % -> 90 % downhill rule."),
    ]),
    ("Root plate and pit", [
        ("sp", "root_plate_diameter", "Root-plate diameter (m)", "float", "Mature equivalent diameter sqrt(length x width)."),
        ("sp", "root_plate_diameter_cv", "Diameter CV", "float", "Coefficient of variation between trees."),
        ("sp", "root_plate_depth", "Root-plate depth (m)", "float", "Mature root-plate (pit) depth."),
        ("sp", "root_plate_depth_cv", "Depth CV", "float", "Coefficient of variation between trees."),
        ("sp", "size_depth_correlation", "Size-depth correlation", "float",
         "Negative: wide plates are shallow (Zadrożny et al. 2017, r = -0.8)."),
        ("sp", "pit_aspect", "Pit width / length", "float", "Pit is wider across the fall line (Zadrożny: 1.54)."),
        ("sp", "pit_shape_exponent", "Pit shape exponent n", "float",
         "Depth profile H(1 - rho^n): 2 = paraboloid, 4 = half-ellipsoid volume (flat floor)."),
        ("sp", "fresh_collapse_scale", "Pit-mound collapse scale", "float",
         "Multiplier on the fresh pit-mound collapse rate (deciduous 0.4)."),
    ]),
]
HILLSLOPE = [
    ("Geometry and run", [
        ("cfg", "length", "Length, ridge to channel (m)", "float", "Horizontal hillslope length."),
        ("cfg", "width", "Width (m)", "float", "Across-slope width (periodic sides)."),
        ("cfg", "slope", "Initial gradient (m/m)", "float", "Planar initial slope; the angle in degrees is shown alongside."),
        ("cfg", "dx", "Grid spacing (m)", "float", "0.25 m resolves root plates well; larger is faster."),
        ("cfg", "years", "Run length (yr)", "float", "Simulated years."),
        ("cfg", "seed", "Random seed", "int", "Same seed = same run."),
        ("cfg", "baselevel_lowering_rate", "Channel lowering (m/yr)", "float", "Base-level fall at the channel."),
    ]),
    ("Soil and bedrock", [
        ("cfg", "soil_model", "Track soil over bedrock", "bool", "Off = unlimited soil."),
        ("cfg", "initial_soil_depth", "Initial soil depth (m)", "float", ""),
        ("cfg", "soil_production_max", "Max soil production (m/yr)", "float", "Bedrock lowering rate on bare rock."),
        ("cfg", "soil_production_scale", "Production depth scale (m)", "float", "P = P0 exp(-h / h0)."),
        ("cfg", "transport_depth_scale", "Creep depth scale (m)", "float", "Creep scales as 1 - exp(-h / h*)."),
        ("cfg", "rock_to_soil_dilation", "Rock/soil density ratio", "float", "Volume gain when rock becomes soil."),
        ("cfg", "bedrock_pluck_fraction", "Bedrock pluck fraction", "float", "Share of root penetration below soil that plucks rock."),
        ("cfg", "thin_soil_depth", "Thin-soil depth (m)", "float", "Root plates thin toward half depth on bare rock (0 = off)."),
    ]),
]
PROCESSES = [
    ("Background creep", [
        ("cfg", "diffusivity", "Diffusivity D (m²/yr)", "float", "Creep, rain splash, bioturbation."),
        ("cfg", "transport_law", "Transport law", ("linear", "nonlinear"), "Nonlinear: D / (1 - (S/Sc)^2)."),
        ("cfg", "critical_slope", "Critical gradient Sc", "float", "Only used by the nonlinear law."),
    ]),
    ("Tree throw", [
        ("cfg", "tree_throw", "Tree throw on", "bool", "Off = diffusion-only control run."),
        ("cfg", "infill_fraction", "Fall-back into pit (flat)", "float", "Share of root-plate soil that falls back into the pit on flat ground."),
        ("cfg", "infill_critical_slope", "No fall-back gradient", "float", "Fall-line gradient where no soil falls back (Norman 1995: ~47°)."),
        ("cfg", "wind_direction_deg", "Wind direction (°, 0 = downslope)", "float", "Prevailing fall azimuth."),
        ("cfg", "wind_concentration", "Wind strength (kappa)", "float", "0 = no wind bias."),
    ]),
    ("Pit-mound collapse", [
        ("cfg", "fresh_diffusivity", "Fresh-soil diffusivity (m²/yr)", "float", "Calibrated to Zadrożny 2017: 0.20."),
        ("cfg", "fresh_decay_time", "Decay time (yr)", "float", "e-folding time of the fresh-soil diffusivity: 3 yr."),
        ("cfg", "fresh_margin", "Disturbed margin (m)", "float", "Fresh zone extends this far beyond pit and mound."),
    ]),
    ("Root growth-decay transport", [
        ("cfg", "root_cycle", "Root cycling on", "bool", "Hoffman & Anderson (2013); adds to the creep diffusivity."),
        ("cfg", "root_biomass", "Root biomass (kg/m²)", "float", ""),
        ("cfg", "root_turnover", "Root turnover (/yr)", "float", ""),
        ("cfg", "root_density", "Root density (kg/m³)", "float", ""),
        ("cfg", "root_beta", "Rooting parameter beta", "float", "Gale & Grigal profile Y = 1 - beta^(100 z)."),
        ("cfg", "a_horizon_depth", "A-horizon depth (m)", "optfloat", "Blank = all roots lift the surface."),
    ]),
]
DISTURBANCE = [
    ("Regime", [
        ("cfg", "disturbance", "Disturbance type", ("none", "fire", "windstorm"), ""),
        ("cfg", "first_disturbance", "First event (yr)", "optfloat", "Blank = random draw from the interval."),
        ("cfg", "disturbance_interval", "Mean return interval (yr)", "float", "0 = only the first event."),
        ("cfg", "disturbance_severity", "Severity (fraction killed)", "float", "Windstorms weight this by tree size."),
    ]),
    ("Windstorm", [
        ("cfg", "storm_uproot_fraction", "Storm uprooting share", "float", "Share of storm-felled trees that uproot."),
        ("cfg", "storm_size_exponent", "Size selectivity", "float", "Kill probability ~ (plate size / mean)^k (Constantine 2012)."),
        ("cfg", "storm_min_age", "Minimum storm age (yr)", "float", "Younger (< ~5 m tall) trees are not felled."),
    ]),
    ("Fire", [
        ("cfg", "snag_fall_rate", "Snag fall rate (/yr)", "float", "Fire-killed trees topple at this rate (Gallaway 2009)."),
        ("cfg", "root_decay_time", "Root decay time (yr)", "float", "Root-plate volume of a dead tree decays as exp(-t / tau)."),
    ]),
]
OUTPUT = [
    ("Animation", [
        ("anim", "frames", "Animation frames", "int", "Frames captured over the run."),
        ("anim", "scale", "Pixel scale", ("2", "3", "4"), "Display size: 320 x 200 times this."),
        ("anim", "fps", "Frames per second", "int", ""),
        ("anim", "scanlines", "CRT scanlines", "bool", ""),
    ]),
    ("Saving", [
        ("anim", "save_results", "Save results after each run", "bool", "events.csv, summary.json, grids and figure."),
        ("anim", "outdir", "Results folder", "str", ""),
    ]),
]
ANIM_DEFAULTS = {"frames": 180, "scale": "3", "fps": 12, "scanlines": True, "save_results": False,
                 "outdir": str(Path.cwd() / "tree_throw_output")}
GUI_CFG_DEFAULTS = {"length": 50.0, "width": 20.0, "slope": 0.35, "years": 3000.0}


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        if text:
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=320, justify="left", font=("TkDefaultFont", 9)).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _unbind_wheel(self, _=None):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(ev)

    def _wheel(self, e):
        step = -1 if e.delta > 0 else 1
        if sys.platform == "win32" or abs(e.delta) >= 120:
            step = int(-e.delta / 120)
        self.canvas.yview_scroll(step, "units")


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------


class TreeThrowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1480x920")
        self.minsize(1100, 700)
        self.fields = {}                    # (target, key) -> (var, kind)
        self.custom_species = None
        self.queue = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self.model = self.frames = self.meta = self.renderer = None
        self.rendered = []
        self.frame_i = 0
        self.playing = False
        self._photo = None
        self._play_job = None
        self._build_style()
        self._build_menu()
        self._build_layout()
        self.load_species("deciduous")
        for k, v in GUI_CFG_DEFAULTS.items():
            self.fields[("cfg", k)][0].set(v)
        self.update_derived()
        self._poll_job = self.after(100, self._poll)
        self.bind("<space>", lambda e: self.toggle_play() if self.focus_get() is not None and
                  not isinstance(self.focus_get(), (ttk.Entry, tk.Entry)) else None)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ layout
    def _build_style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"), padding=6)
        st.configure("Group.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        st.configure("Derived.TLabel", font=("TkFixedFont", 9))

    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=False)
        f.add_command(label="Load species JSON…", command=self.load_species_json)
        f.add_command(label="Save species JSON…", command=self.save_species_json)
        f.add_separator()
        f.add_command(label="Save results…", command=self.save_results_dialog)
        f.add_command(label="Save animation GIF…", command=self.save_gif)
        f.add_command(label="Save frame PNG…", command=self.save_png)
        f.add_separator()
        f.add_command(label="Quit", command=self.on_close)
        m.add_cascade(label="File", menu=f)
        h = tk.Menu(m, tearoff=False)
        h.add_command(label="Quick guide", command=self.show_help)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)

    def _build_layout(self):
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=4)
        right = ttk.Frame(pw, padding=4)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        # species selector
        top = ttk.Frame(left)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Forest type", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.species_var = tk.StringVar()
        cb = ttk.Combobox(top, textvariable=self.species_var, state="readonly", width=24,
                          values=list(tt.SPECIES) + [CUSTOM])
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", self.on_species_selected)
        ttk.Button(top, text="Reset", command=lambda: self.load_species(self.species_var.get())).pack(side="left")
        self.species_desc = ttk.Label(left, wraplength=400, justify="left", foreground="#444")
        self.species_desc.pack(fill="x", pady=(0, 4))

        nb = ttk.Notebook(left, width=430)
        nb.pack(fill="both", expand=True)
        for title, spec in (("Forest", FOREST), ("Hillslope", HILLSLOPE), ("Processes", PROCESSES),
                            ("Disturbance", DISTURBANCE), ("Output", OUTPUT)):
            sf = ScrollFrame(nb)
            nb.add(sf, text=title)
            if title == "Forest":
                self._build_derived(sf.inner)
            self._build_groups(sf.inner, spec)
            if title == "Hillslope":
                self._build_size_info(sf.inner)
            if title == "Disturbance":
                self._build_disturbance_presets(sf.inner)
        self._build_run_bar(left)

        rnb = ttk.Notebook(right)
        rnb.pack(fill="both", expand=True)
        self.right_nb = rnb
        self._build_player(rnb)
        self._build_results_tab(rnb)
        self._build_summary_tab(rnb)
        self._build_log_tab(rnb)

    def _build_groups(self, parent, spec):
        for gname, items in spec:
            lf = ttk.LabelFrame(parent, text=gname, style="Group.TLabelframe", padding=6)
            lf.pack(fill="x", padx=4, pady=4)
            lf.columnconfigure(1, weight=1)
            for r, (target, key, label, kind, tip) in enumerate(items):
                lab = ttk.Label(lf, text=label)
                lab.grid(row=r, column=0, sticky="w", pady=1)
                if kind == "bool":
                    var = tk.BooleanVar()
                    w = ttk.Checkbutton(lf, variable=var)
                elif isinstance(kind, tuple):
                    var = tk.StringVar()
                    w = ttk.Combobox(lf, textvariable=var, values=kind, state="readonly", width=12)
                else:
                    var = tk.StringVar()
                    w = ttk.Entry(lf, textvariable=var, width=14 if kind != "str" else 30)
                w.grid(row=r, column=1, sticky="e", padx=(6, 0), pady=1)
                if key == "slope":
                    self.slope_deg = ttk.Label(lf, text="", foreground="#666")
                    self.slope_deg.grid(row=r, column=2, sticky="w", padx=4)
                    var.trace_add("write", lambda *a: self._update_slope_deg())
                if target == "sp" or key in ("slope", "infill_fraction", "infill_critical_slope", "length",
                                            "width", "dx"):
                    var.trace_add("write", lambda *a: self._schedule_derived())
                ToolTip(lab, tip)
                ToolTip(w, tip)
                self.fields[(target, key)] = (var, kind)
                if target == "anim":
                    var.set(ANIM_DEFAULTS[key])
                elif target == "cfg":
                    self._set_var(var, kind, getattr(tt.HillslopeConfig(), key))

    def _build_derived(self, parent):
        lf = ttk.LabelFrame(parent, text="Derived for these settings", style="Group.TLabelframe", padding=6)
        lf.pack(fill="x", padx=4, pady=4)
        self.derived = ttk.Label(lf, text="", style="Derived.TLabel", justify="left")
        self.derived.pack(fill="x")

    def _build_size_info(self, parent):
        lf = ttk.LabelFrame(parent, text="Run size", style="Group.TLabelframe", padding=6)
        lf.pack(fill="x", padx=4, pady=4)
        self.size_info = ttk.Label(lf, text="", style="Derived.TLabel", justify="left")
        self.size_info.pack(fill="x")

    def _build_disturbance_presets(self, parent):
        lf = ttk.LabelFrame(parent, text="Presets", style="Group.TLabelframe", padding=6)
        lf.pack(fill="x", padx=4, pady=4)

        def fire():
            self._setf("cfg", "disturbance", "fire")
            self._setf("cfg", "first_disturbance", round(0.5 * self._getf("cfg", "years", 3000.0)))
            self._setf("cfg", "disturbance_interval", 0.0)
            self._setf("cfg", "disturbance_severity", 1.0)

        def storms():
            self._setf("cfg", "disturbance", "windstorm")
            self._setf("cfg", "first_disturbance", "")
            self._setf("cfg", "disturbance_interval", 100.0)
            self._setf("cfg", "disturbance_severity", 0.2)

        def none():
            self._setf("cfg", "disturbance", "none")
        for text, fn in (("Crown fire at mid-run", fire), ("Storm every ~100 yr", storms), ("No disturbance", none)):
            ttk.Button(lf, text=text, command=fn).pack(fill="x", pady=1)

    def _build_run_bar(self, parent):
        bar = ttk.Frame(parent, padding=(0, 6, 0, 0))
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="▶  Run simulation", style="Run.TButton", command=self.start_run)
        self.run_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(bar, text="■ Stop", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(parent, mode="determinate", maximum=1.0)
        self.progress.pack(fill="x", pady=(6, 2))
        self.status = ttk.Label(parent, text="Ready.", foreground="#225")
        self.status.pack(fill="x")

    def _build_player(self, nb):
        tab = tk.Frame(nb, background="#101018")
        nb.add(tab, text="Retro view")
        self.screen = tk.Label(tab, background="#101018", text="Run a simulation to start the animation.",
                               foreground="#8f8", font=("Courier", 14, "bold"))
        self.screen.pack(fill="both", expand=True)
        ctl = ttk.Frame(tab, padding=(4, 4, 4, 0))
        ctl.pack(fill="x")
        ctl2 = ttk.Frame(tab, padding=(4, 2, 4, 2))
        ctl2.pack(fill="x")
        b = {}
        for name, text, cmd in (("first", "⏮", lambda: self.goto(0)), ("back", "◀", lambda: self.step(-1)),
                                ("play", "▶ Play", self.toggle_play), ("fwd", "▶|", lambda: self.step(1)),
                                ("last", "⏭", lambda: self.goto(len(self.rendered) - 1))):
            b[name] = ttk.Button(ctl, text=text, width=7 if name == "play" else 3, command=cmd)
            b[name].pack(side="left", padx=1)
        self.play_btn = b["play"]
        self.scrub = tk.Scale(ctl, from_=0, to=0, orient="horizontal", showvalue=False, length=300,
                              command=lambda v: self.goto(int(float(v)), from_scale=True))
        self.scrub.pack(side="left", fill="x", expand=True, padx=6)
        self.frame_lbl = ttk.Label(ctl, text="", width=24)
        self.frame_lbl.pack(side="left")
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl2, text="Loop", variable=self.loop_var).pack(side="left", padx=4)
        ttk.Label(ctl2, text="Speed (fps)").pack(side="left", padx=(12, 2))
        ttk.Spinbox(ctl2, from_=1, to=60, width=4, textvariable=self.fields[("anim", "fps")][0]
                    if ("anim", "fps") in self.fields else None).pack(side="left")
        ttk.Label(ctl2, text="Pixel scale").pack(side="left", padx=(12, 2))
        sc = ttk.Combobox(ctl2, values=("2", "3", "4"), width=3, state="readonly",
                          textvariable=self.fields[("anim", "scale")][0] if ("anim", "scale") in self.fields else None)
        sc.pack(side="left")
        sc.bind("<<ComboboxSelected>>", lambda e: self.goto(self.frame_i))
        ttk.Checkbutton(ctl2, text="CRT scanlines", variable=self.fields[("anim", "scanlines")][0]
                        if ("anim", "scanlines") in self.fields else None,
                        command=lambda: self.goto(self.frame_i)).pack(side="left", padx=12)
        ttk.Button(ctl2, text="Save PNG…", command=self.save_png).pack(side="right", padx=2)
        ttk.Button(ctl2, text="Save GIF…", command=self.save_gif).pack(side="right", padx=2)
        legend = ("Side view: transect with soil (brown) over bedrock (grey); trees are the simulated stand, "
                  "toppling trees show uprootings (log + root wad). Plan view: pit-mound relief "
                  "(blue pits, yellow mounds), canopies, flashing new uprootings with fall direction. "
                  "HUD: live model statistics. Relief and soil are vertically exaggerated as labelled.")
        ttk.Label(tab, text=legend, wraplength=900, foreground="#ccc", background="#101018",
                  justify="left").pack(fill="x", padx=6, pady=(0, 4))

    def _build_results_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Results")
        self.fig = Figure(figsize=(13, 8), dpi=90)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=tab)
        NavigationToolbar2Tk(self.fig_canvas, tab).update()
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_summary_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Summary")
        self.summary_txt = scrolledtext.ScrolledText(tab, font=("TkFixedFont", 10), wrap="word")
        self.summary_txt.pack(fill="both", expand=True)

    def _build_log_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Log")
        self.log_txt = scrolledtext.ScrolledText(tab, font=("TkFixedFont", 9), height=10)
        self.log_txt.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- fields
    @staticmethod
    def _set_var(var, kind, value):
        if kind == "bool":
            var.set(bool(value))
        elif value is None:
            var.set("")
        elif isinstance(value, float):
            var.set(f"{value:.6g}")
        else:
            var.set(str(value))

    def _setf(self, target, key, value):
        var, kind = self.fields[(target, key)]
        self._set_var(var, kind, value)

    def _getf(self, target, key, default=None):
        try:
            return self._parse(target, key)
        except ValueError:
            return default

    def _parse(self, target, key):
        var, kind = self.fields[(target, key)]
        v = var.get()
        if kind == "bool":
            return bool(v)
        if isinstance(kind, tuple) or kind == "str":
            return v
        s = str(v).strip()
        if kind == "optfloat":
            return None if s == "" else float(s)
        if kind == "int":
            return int(float(s))
        return float(s)

    def log(self, msg):
        self.log_txt.insert("end", time.strftime("%H:%M:%S  ") + msg + "\n")
        self.log_txt.see("end")

    # ---------------------------------------------------------------- species
    def on_species_selected(self, _=None):
        key = self.species_var.get()
        if key == CUSTOM:
            self.load_species_json()
        else:
            self.load_species(key)

    def load_species(self, key):
        if key == CUSTOM and self.custom_species is not None:
            sp = self.custom_species
        else:
            key = key if key in tt.SPECIES else "deciduous"
            sp = tt.SPECIES[key]
        self.species_var.set(key)
        self.species_desc.config(text=sp.description)
        for (target, k), (var, kind) in self.fields.items():
            if target == "sp":
                self._set_var(var, kind, getattr(sp, k))
        self.update_derived()

    def current_species(self):
        base = self.custom_species if (self.species_var.get() == CUSTOM and self.custom_species) \
            else tt.SPECIES.get(self.species_var.get(), tt.SPECIES["deciduous"])
        changes = {k: self._parse("sp", k) for (t, k) in self.fields if t == "sp"}
        sp = replace(base, **changes)
        sp.validate()
        return sp

    def current_config(self):
        kw = {k: self._parse("cfg", k) for (t, k) in self.fields if t == "cfg"}
        kw["n_snapshots"] = 6
        kw["record_interval"] = max(kw.get("years", 1000) / 300.0, 1.0)
        cfg = tt.HillslopeConfig(**kw)
        cfg.validate()
        return cfg

    def load_species_json(self):
        path = filedialog.askopenfilename(title="Load species JSON", filetypes=[("JSON", "*.json"), ("All", "*")])
        if not path:
            if self.custom_species is None:
                self.species_var.set("deciduous")
            return
        try:
            self.custom_species = tt.load_species_file(path)
        except Exception as e:      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Could not load species:\n{e}")
            return
        self.species_var.set(CUSTOM)
        self.load_species(CUSTOM)
        self.log(f"Loaded species from {path}")

    def save_species_json(self):
        try:
            sp = self.current_species()
        except Exception as e:      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Invalid species parameters:\n{e}")
            return
        path = filedialog.asksaveasfilename(title="Save species JSON", defaultextension=".json",
                                            initialfile=f"{sp.key}_custom.json")
        if path:
            Path(path).write_text(json.dumps(asdict(sp), indent=2), encoding="utf-8")
            self.log(f"Saved species to {path}")

    # ---------------------------------------------------------------- derived
    def _schedule_derived(self):
        if getattr(self, "_derived_job", None):
            self.after_cancel(self._derived_job)
        self._derived_job = self.after(400, self.update_derived)

    def _update_slope_deg(self):
        try:
            s = float(self.fields[("cfg", "slope")][0].get())
            self.slope_deg.config(text=f"= {math.degrees(math.atan(s)):.1f}°")
        except (ValueError, KeyError):
            self.slope_deg.config(text="")

    def update_derived(self):
        self._derived_job = None
        if not hasattr(self, "derived"):
            return
        try:
            sp = self.current_species()
            slope = self._parse("cfg", "slope")
            r = tt.stand_rates(sp)
            e = tt.expected_tree_throw(sp, slope, self._parse("cfg", "infill_fraction"),
                                       self._parse("cfg", "infill_critical_slope"), n_samples=20000)
            txt = (f"Deaths            {r['deaths_per_ha_yr']:7.2f} /ha/yr\n"
                   f"Uprooting share   {100 * r['uprooting_share_of_deaths']:7.0f} % of deaths\n"
                   f"Uprootings        {r['uprootings_per_ha_yr']:7.2f} /ha/yr\n"
                   f"Mean age at death {r['mean_age_at_death']:7.0f} yr\n"
                   f"Excavated volume  {e['mean_event_volume_m3']:7.2f} m³ per event\n"
                   f"Displaced volume  {e['mean_displaced_volume_m3']:7.2f} m³ per event\n"
                   f"Tree-throw D      {e['diffusivity_m2_per_yr']:7.1e} m²/yr at S = {slope:.2f}\n"
                   f"Surface turnover  {e['surface_turnover_yr']:7.0f} yr")
        except Exception as ex:     # noqa: BLE001
            txt = f"(invalid parameters: {ex})"
        self.derived.config(text=txt)
        try:
            L, W, dx = (self._parse("cfg", k) for k in ("length", "width", "dx"))
            cells = (L / dx + 1) * (W / dx)
            trees = self._parse("sp", "density") * L * W / 1e4
            yrs = self._parse("cfg", "years")
            est = cells * yrs * 1.2e-7 + trees * yrs * 2e-6
            self.size_info.config(text=f"Grid {L / dx + 1:.0f} x {W / dx:.0f} = {cells:,.0f} cells\n"
                                       f"Stand ~{trees:.0f} trees, {yrs:.0f} yr\n"
                                       f"Rough run time ~{max(est, 1):.0f} s")
        except Exception:           # noqa: BLE001
            pass

    # ---------------------------------------------------------------- run
    def start_run(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            sp = self.current_species()
            cfg = self.current_config()
            n_frames = max(int(self._parse("anim", "frames")), 10)
        except Exception as e:      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Please check the parameters:\n{e}")
            return
        self.stop_playback()
        self.stop_event.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.status.config(text="Starting…")
        self.log(f"Run: {sp.key}, L={cfg.length} m, W={cfg.width} m, S={cfg.slope}, {cfg.years:.0f} yr, "
                 f"disturbance={cfg.disturbance}")

        def work():
            try:
                t0 = time.time()
                model, frames, meta = tr.run_with_frames(
                    sp, cfg, n_frames=n_frames,
                    progress=lambda f, m: self.queue.put(("progress", 0.9 * f, m)), stop=self.stop_event)
                self.queue.put(("progress", 0.9, "rendering animation…"))
                rend = tr.RetroRenderer(frames, meta)
                imgs = []
                for k in range(len(frames)):
                    imgs.append(rend.render(k))
                    if k % 10 == 0:
                        self.queue.put(("progress", 0.9 + 0.1 * k / len(frames), f"rendering frame {k}"))
                self.queue.put(("done", model, frames, meta, rend, imgs, time.time() - t0))
            except Exception:       # noqa: BLE001
                self.queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop_run(self):
        self.stop_event.set()
        self.status.config(text="Stopping after the current step…")

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    self.progress["value"] = msg[1]
                    self.status.config(text=msg[2])
                elif msg[0] == "done":
                    self._on_done(*msg[1:])
                elif msg[0] == "error":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status.config(text="Run failed — see Log.")
                    self.log(msg[1])
                    messagebox.showerror(APP_TITLE, msg[1].strip().splitlines()[-1])
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    def _on_done(self, model, frames, meta, rend, imgs, secs):
        self.model, self.frames, self.meta, self.renderer, self.rendered = model, frames, meta, rend, imgs
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress["value"] = 1.0
        stopped = " (stopped early)" if self.stop_event.is_set() else ""
        self.status.config(text=f"Done{stopped}: {model.time:.0f} yr, {model.n_events} uprootings, "
                                f"{len(frames)} frames, {secs:.1f} s.")
        self.log(f"Finished{stopped} in {secs:.1f} s: {model.n_events} uprootings, {len(frames)} frames.")
        self.scrub.config(to=max(len(imgs) - 1, 0))
        self.goto(0)
        self.right_nb.select(0)
        self.draw_results()
        self.write_summary()
        if self._parse("anim", "save_results"):
            self.save_results(Path(self._parse("anim", "outdir")))
        self.toggle_play(force=True)

    # ---------------------------------------------------------------- results
    def draw_results(self):
        self.fig.clear()
        try:
            tt.draw_run_figure(self.model, self.fig)
            self.fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.07, wspace=0.55, hspace=0.38)
        except Exception:           # noqa: BLE001
            self.log(traceback.format_exc())
        self.fig_canvas.draw_idle()

    def write_summary(self):
        s = self.model.summary()
        sp = self.model.sp
        lines = [
            f"{sp.name}",
            f"{s['years_simulated']:.0f} yr on a {s['grid']['hillslope_length_m']:.0f} x {s['grid']['width_m']:.0f} m "
            f"hillslope, initial gradient {self.model.cfg.slope:.2f}", "",
            "Forest",
            f"  trees in stand            {s['trees_in_stand']}",
            f"  deaths                    {s['tree_deaths']}  ({s['death_rate_per_ha_yr']:.2f} /ha/yr)",
            f"  uprootings                {s['uprooting_events']}  ({s['uprooting_rate_per_ha_yr']:.2f} /ha/yr, "
            f"{100 * s['uprooting_share_of_deaths']:.0f} % of deaths)",
            f"  by cause                  {s['uprooting_events_by_cause']}",
            f"  disturbances              {s['disturbances'] or 'none'}", "",
            "Soil transport (mean downslope flux per unit width)",
            f"  uprooting                 {s['mean_flux_tree_throw_m2_yr']:.2e} m²/yr",
            f"  pit-mound collapse        {s['mean_flux_pit_mound_collapse_m2_yr']:.2e} m²/yr  "
            "(negative = mound soil slumping back upslope into pits)",
            f"  background creep          {s['mean_flux_diffusion_m2_yr']:.2e} m²/yr",
            f"  tree-throw share          {100 * s['tree_throw_share_of_flux']:.0f} %",
            f"  tree-throw diffusivity    {s['effective_D_tree_throw_m2_yr']:.2e} m²/yr "
            f"(analytic {s['expected_D_tree_throw_m2_yr']:.2e})",
            f"  sediment yield            {s['sediment_yield_m3_per_m_width_yr'] * 1e4:.2f} cm³/cm/yr", "",
            "Surface",
            f"  mean soil depth           {s['final_mean_soil_depth_m']:.2f} m",
            f"  pit-mound roughness       {100 * s['final_roughness_m']:.1f} cm RMS",
            f"  ridge lowering            {s['max_ridge_lowering_m']:.2f} m",
            f"  soil from root plucking   {s['soil_produced_by_root_plucking_m3']:.2f} m³", "",
            "Species parameters",
        ] + [f"  {k:24s} {v}" for k, v in asdict(sp).items() if k not in ("name", "description")]
        self.summary_txt.delete("1.0", "end")
        self.summary_txt.insert("1.0", "\n".join(lines))

    def save_results(self, outdir: Path):
        if self.model is None:
            return
        d = outdir / self.model.sp.key
        try:
            self.model.save(d)
            self.fig.savefig(d / "hillslope_evolution.png", dpi=120)
            self.log(f"Saved results to {d}")
        except Exception as e:      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Could not save results:\n{e}")

    def save_results_dialog(self):
        if self.model is None:
            messagebox.showinfo(APP_TITLE, "Run a simulation first.")
            return
        d = filedialog.askdirectory(title="Results folder")
        if d:
            self.save_results(Path(d))

    def save_gif(self):
        if self.renderer is None:
            messagebox.showinfo(APP_TITLE, "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(title="Save animation", defaultextension=".gif",
                                            initialfile=f"tree_throw_{self.model.sp.key}.gif")
        if not path:
            return
        self.status.config(text="Writing GIF…")
        self.update_idletasks()
        try:
            tr.save_gif(self.renderer, path, scale=int(self._parse("anim", "scale")),
                        scanlines=self._parse("anim", "scanlines"), fps=int(self._parse("anim", "fps")))
            self.status.config(text=f"Saved {path}")
            self.log(f"Saved animation {path}")
        except Exception as e:      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Could not write GIF (needs Pillow):\n{e}")

    def save_png(self):
        if not self.rendered:
            messagebox.showinfo(APP_TITLE, "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(title="Save frame", defaultextension=".png",
                                            initialfile=f"tree_throw_frame_{self.frame_i:03d}.png")
        if path:
            from PIL import Image
            Image.fromarray(self._display_rgb(self.frame_i)).save(path)
            self.log(f"Saved frame {path}")

    # ---------------------------------------------------------------- player
    def _display_rgb(self, i):
        return tr.upscale(self.rendered[i], int(self._parse("anim", "scale")), self._parse("anim", "scanlines"))

    def _show(self, i):
        rgb = self._display_rgb(i)
        try:
            self._photo = tk.PhotoImage(data=tr.to_ppm(rgb), format="PPM")
        except tk.TclError:
            from PIL import Image, ImageTk
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.screen.config(image=self._photo, text="")
        f = self.frames[i]
        self.frame_lbl.config(text=f"frame {i + 1}/{len(self.rendered)}   year {f.t:.0f}")

    def goto(self, i, from_scale=False):
        if not self.rendered:
            return
        self.frame_i = int(np.clip(i, 0, len(self.rendered) - 1))
        if not from_scale:
            self.scrub.set(self.frame_i)
        self._show(self.frame_i)

    def step(self, d):
        self.stop_playback()
        self.goto(self.frame_i + d)

    def toggle_play(self, force=False):
        if not self.rendered:
            return
        if self.playing and not force:
            self.stop_playback()
            return
        self.playing = True
        self.play_btn.config(text="❚❚ Pause")
        if self.frame_i >= len(self.rendered) - 1:
            self.goto(0)
        self._tick()

    def stop_playback(self):
        self.playing = False
        if self._play_job:
            self.after_cancel(self._play_job)
            self._play_job = None
        self.play_btn.config(text="▶ Play")

    def _tick(self):
        if not self.playing:
            return
        nxt = self.frame_i + 1
        if nxt >= len(self.rendered):
            if self.loop_var.get():
                nxt = 0
            else:
                self.stop_playback()
                return
        self.goto(nxt)
        fps = max(int(self._getf("anim", "fps", 12) or 12), 1)
        delay = 2000 if nxt == len(self.rendered) - 1 else int(1000 / fps)
        self._play_job = self.after(delay, self._tick)

    # ---------------------------------------------------------------- misc
    def show_help(self):
        messagebox.showinfo(APP_TITLE, (
            "1. Pick a forest type (Forest tab) and adjust its parameters; the 'Derived' box shows the "
            "mortality, uprooting rate and expected tree-throw diffusivity they imply.\n"
            "2. Set the hillslope size, slope and run length (Hillslope tab); the 'Run size' box estimates "
            "the run time.\n"
            "3. Optionally add a fire or windstorm regime (Disturbance tab).\n"
            "4. Press Run.  The Retro view plays the simulated hillslope as a 1980s pixel animation; Results "
            "shows the scientific figure and Summary the key numbers.\n"
            "5. File menu: save results, the GIF animation or a frame; load or save custom species."))

    def on_close(self):
        self.stop_event.set()
        self.stop_playback()
        for job in (getattr(self, "_poll_job", None), getattr(self, "_derived_job", None)):
            if job:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self.destroy()


def main():
    app = TreeThrowApp()
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
