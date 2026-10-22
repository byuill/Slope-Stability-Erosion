"""Particle-size input and reproducible distributions shared by both models."""
from dataclasses import dataclass
import math
from statistics import NormalDist
import numpy as np


@dataclass(frozen=True)
class SizeDistribution:
    mode: str
    values: tuple

    @property
    def representative(self):
        return float(np.median(self.values))

    @property
    def class_count(self):
        return len(self.values) if self.mode == "equal" else 1

    def quantile(self, probability):
        """Two lognormal branches anchored at the specified 16/50/84 percentiles."""
        low, median, high = self.values
        z = NormalDist().inv_cdf(float(probability))
        spread = math.log(median/low) if z < 0 else math.log(high/median)
        return median * math.exp(z * spread / NormalDist().inv_cdf(.84))

    def sample(self, count, rng):
        if count < 1:
            raise ValueError("Particle count must be positive.")
        if self.mode == "single":
            sizes = np.full(count, self.values[0])
        elif self.mode == "equal":
            if count % self.class_count:
                raise ValueError(f"Particle count must be a multiple of {self.class_count} "
                                 "for equal numbers of the semicolon-separated sizes.")
            sizes = np.tile(self.values, count//self.class_count).astype(float)
        else:
            # Stratified quantiles reduce finite-sample error without infinite tails.
            sizes = np.array([self.quantile((i+.5)/count) for i in range(count)])
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("The size percentiles produce an unrepresentable size range.")
        rng.shuffle(sizes)
        return sizes

    def next_size(self, rng, packed):
        if self.mode == "equal":
            counts = [sum(math.isclose(p["size"], value, rel_tol=1e-10, abs_tol=0.)
                          for p in packed) for value in self.values]
            return self.values[int(np.argmin(counts))]
        if self.mode == "percentiles":
            return self.quantile(float(rng.uniform(.000001, .999999)))
        return self.values[0]


def parse_size_input(value):
    text = str(value).strip()
    if "," in text and ";" in text:
        raise ValueError("Use commas for D16,D50,D84 or semicolons for equal-count sizes; do not mix separators.")
    separator = ";" if ";" in text else "," if "," in text else None
    parts = text.split(separator) if separator else [text]
    try:
        values = tuple(float(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("Enter one diameter, D16,D50,D84, or sizes separated by semicolons (metres).") from exc
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("All particle sizes must be positive, finite values in metres.")
    if separator == ",":
        if len(values) != 3 or not values[0] <= values[1] <= values[2]:
            raise ValueError("Comma input requires three ascending values: D16,D50,D84.")
        mode = "percentiles"
    elif separator == ";":
        if len(set(values)) != len(values):
            raise ValueError("Semicolon sizes must be distinct; each size receives an equal particle count.")
        mode = "equal"
    else:
        mode = "single"
    return SizeDistribution(mode, values)


def format_duration(seconds):
    if seconds is None:
        return "estimating…"
    seconds = max(0, int(math.ceil(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ProgressEstimate:
    """Estimate remaining wall time from completed steps, smoothing recent speed."""
    def __init__(self):
        self.start = self.last_time = None
        self.last_step = 0
        self.rate = None

    def update(self, step, total, now):
        if self.start is None:
            self.start = self.last_time = now
            self.last_step = step
        elapsed = max(0., now-self.start)
        interval = now-self.last_time
        if interval > 0 and step > self.last_step:
            rate = (step-self.last_step)/interval
            alpha = 1-math.exp(-interval/5.)
            self.rate = rate if self.rate is None else self.rate + alpha*(rate-self.rate)
            self.last_time, self.last_step = now, step
        eta = (max(0, total-step)/self.rate if self.rate and elapsed >= .2 else None)
        if step >= total:
            eta = 0.
        return elapsed, eta
