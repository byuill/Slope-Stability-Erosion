"""Select existing solver states for animation without changing integration dt."""
import math


def validate_recording(s):
    if s.FRAME_INTERVAL_MODE not in ("steps", "seconds"):
        raise ValueError("Frame interval units must be steps or seconds.")
    if not isinstance(s.FRAME_INTERVAL, int) or s.FRAME_INTERVAL < 1:
        raise ValueError("Frame interval in steps must be a positive integer.")
    if not math.isfinite(s.FRAME_INTERVAL_SECONDS) or s.FRAME_INTERVAL_SECONDS <= 0:
        raise ValueError("Frame interval in seconds must be positive and finite.")
    stop = s.ANIMATION_STOP_TIME
    if (not math.isfinite(s.ANIMATION_START_TIME) or s.ANIMATION_START_TIME < 0
            or stop is not None and (not math.isfinite(stop) or stop <= s.ANIMATION_START_TIME)):
        raise ValueError("Animation times require 0 <= start < stop; leave stop blank for the full run.")


class FrameSchedule:
    """Nearest actual state to targets; midpoint ties select the earlier state.

    The caller supplies the already-chosen next computational time. Thus seconds
    recording never forces extra solver steps. Multiple targets landing on one
    state are coalesced. Both window endpoints are sampled by nearest state.
    """
    def __init__(self, s, duration):
        validate_recording(s)
        self.mode = s.FRAME_INTERVAL_MODE
        self.steps = s.FRAME_INTERVAL
        self.seconds = s.FRAME_INTERVAL_SECONDS
        self.start = s.ANIMATION_START_TIME
        self.stop = duration if s.ANIMATION_STOP_TIME is None else min(duration, s.ANIMATION_STOP_TIME)
        if self.start > duration:
            raise ValueError("Animation start time is after the simulation ends.")
        self.duration = duration
        self.target_index = 0
        self.boundaries = [self.start, self.stop]
        self.last_saved_step = None

    def due(self, step, time, next_time=None):
        horizon = time if next_time is None else .5*(time+next_time)
        tol = 1e-12*max(1., self.duration)
        selected = False
        if self.mode == "seconds":
            target = self.start+self.target_index*self.seconds
            if target <= self.stop+tol and target <= horizon+tol:
                selected = True
                self.target_index = max(self.target_index+1,
                                        int(math.floor((horizon-self.start+tol)/self.seconds))+1)
        elif self.start-tol <= time <= self.stop+tol and step % self.steps == 0:
            selected = True
        while self.boundaries and self.boundaries[0] <= horizon+tol:
            selected = True
            self.boundaries.pop(0)
        if selected and self.last_saved_step != step:
            self.last_saved_step = step
            return True
        return False

    def contains(self, time):
        return self.start-1e-12 <= time <= self.stop+1e-12

    def metadata(self):
        return dict(interval_mode=self.mode, interval_steps=self.steps,
                    interval_seconds=self.seconds, start_time=self.start, stop_time=self.stop,
                    selection="nearest solver state; midpoint ties use earlier state")
