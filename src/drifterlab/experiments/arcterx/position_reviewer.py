"""Matplotlib UI for explicit point decisions on a frozen candidate queue."""

import json
from pathlib import Path
import textwrap

import numpy as np
import xarray as xr

from drifterlab.qc.flags import EARTH_RADIUS_M
from drifterlab.qc.position import position_valid
from .position_review import (Decisions, apply_decisions, native_frame, edge_arrays,
                              master_identity, utc_string, write_json, load_suggestions)
from .jump_recovery import RECOVERY_CONFIG, recover_region


class PositionReviewer:
    def __init__(self, master: Path, review: Path, directory: Path, state: dict, queue, edges):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.plt = plt
        self.master = xr.open_zarr(master, consolidated=True, chunks=None)
        try:
            if master_identity(self.master, master) != state["master"]:
                raise ValueError("Master differs from the saved candidate queue")
            if str(review.resolve()) != state["review_file"]:
                raise ValueError("Review CSV differs from the saved candidate queue")
            self.decisions = Decisions(review)
            platforms = list(map(str, self.master.platform_code.values))
            if {key[0] for key in self.decisions.rows} - set(platforms):
                raise ValueError("Review CSV contains unknown platforms")
            self.platform_indices = {p: i for i, p in enumerate(platforms)}
            self.state, self.queue, self.edges = state, queue.reset_index(drop=True), edges
            self.suggestions = load_suggestions(state)
            self.session_path = directory / "session.json"
            self.cursor, self.selected = 0, None
            if self.session_path.exists():
                session = json.loads(self.session_path.read_text(encoding="utf-8"))
                if session.get("review_iteration") == state["review_iteration"]:
                    self.cursor = min(max(0, int(session["cursor"])), max(0, len(queue) - 1))
                    self.selected = session.get("selected_row")
            self.cached_platform = None
            self.figure, (self.map_ax, self.speed_ax) = plt.subplots(
                1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.2, 1]})
            self.figure.subplots_adjust(left=.06, right=.97, top=.89, bottom=.54, wspace=.3)
            # K/R/S/Q overlap Matplotlib's scale/home/save/quit shortcuts.
            handler = getattr(self.figure.canvas.manager, "key_press_handler_id", None)
            if handler is not None:
                self.figure.canvas.mpl_disconnect(handler)
            self.reason_box = TextBox(self.figure.add_axes((.12, .20, .82, .04)), "Reason ")
            self.note_box = TextBox(self.figure.add_axes((.12, .14, .82, .04)), "Note ")
            self.buttons = []
            for index, (label, key) in enumerate([("Keep [K]", "k"), ("Reject [R]", "r"),
                        ("Uncertain [U]", "u"), ("Accept high [A]", "a"), ("Prev [P]", "p"), ("Next [N]", "n"),
                        ("Save [S]", "s"), ("Quit [Q]", "q")]):
                button = Button(self.figure.add_axes((.06 + index * .114, .065, .107, .045)), label)
                button.on_clicked(lambda event, k=key: self.action(k))
                self.buttons.append(button)
            self.info = self.figure.text(.06, .39, "", fontsize=9, va="bottom", family="monospace")
            self.recovery_info = self.figure.text(.06, .255, "", fontsize=9, va="bottom", family="monospace")
            self.status = self.figure.text(.06, .02, "", fontsize=9)
            self.figure.canvas.mpl_connect("key_press_event", self.on_key)
            self.figure.canvas.mpl_connect("button_press_event", self.on_click)
            self.figure.canvas.mpl_connect("close_event", self.on_close)
            self.closed = False
            self.draw()
        except Exception:
            self.master.close()
            raise

    def load_candidate(self):
        candidate = self.queue.iloc[self.cursor]
        platform = str(candidate.platform_code)
        if platform != self.cached_platform:
            index = self.platform_indices[platform]
            self.native = native_frame(self.master, index)
            self.references = []
            if "time_interp" in self.master:
                n = int(self.master.n_obs_interp.values[index])
                names = ["time_interp", "lon_interp_30m", "lat_interp_30m", "lon_interp_60m", "lat_interp_60m"]
                row = self.master[[name for name in names if name in self.master]].isel(
                    trajectory=index, obs_interp=slice(0, n)).load()
                for method in ("30m", "60m"):
                    if f"lon_interp_{method}" in row and f"lat_interp_{method}" in row:
                        self.references.append((method, row.time_interp.values,
                                                row[f"lon_interp_{method}"].values,
                                                row[f"lat_interp_{method}"].values))
            self.cached_platform = platform
        self.frame = apply_decisions(self.native, self.decisions)
        lo, hi = int(candidate.context_start), int(candidate.context_end)
        # Retain rejected points for selection/reversal, including after queue regeneration.
        self.visible = np.arange(lo, hi + 1)
        self.selectable = self.visible[(self.frame.position_valid & self.frame.time.notna()).to_numpy()[self.visible]]
        if self.selected not in self.selectable:
            preferred = int(candidate.selected_row)
            self.selected = preferred if preferred in self.selectable else int(self.selectable[0])
        return candidate

    def sync_text(self, candidate):
        point = self.frame.iloc[self.selected]
        self.reason_box.set_val(point.manual_review_reason or f"{candidate.classification}: {candidate.reason}")
        self.note_box.set_val(point.manual_review_note)

    def draw(self):
        if self.queue.empty:
            self.figure.suptitle("No >3 m/s candidate regions remain in this queue")
            self.status.set_text("Q: save and quit. Uncertain decisions remain valid and flagged in the output.")
            self.figure.canvas.draw_idle()
            return
        candidate = self.load_candidate()
        point = self.frame.iloc[self.selected]
        native = self.frame.iloc[self.visible]
        lon0, lat0 = float(point.lon_qc), float(point.lat_qc)

        def xy(lon, lat):
            delta = (np.asarray(lon) - lon0 + 180) % 360 - 180
            return (EARTH_RADIUS_M * np.radians(delta) * np.cos(np.radians(lat0)) / 1000,
                    EARTH_RADIUS_M * np.radians(np.asarray(lat) - lat0) / 1000)

        ax, speed_ax = self.map_ax, self.speed_ax
        ax.clear()
        speed_ax.clear()
        x, y = xy(native.lon_qc, native.lat_qc)
        valid = native.position_valid.to_numpy() & native.time.notna().to_numpy()
        ax.plot(np.where(valid, x, np.nan), np.where(valid, y, np.nan), "o-", color=".65",
                ms=3, lw=.8, label="Supplied native QC")
        indices, speeds, _ = edge_arrays(self.frame, "surviving")
        local = indices[(indices >= self.visible[0]) & (indices <= self.visible[-1])]
        lx, ly = xy(self.frame.lon_qc.iloc[local], self.frame.lat_qc.iloc[local])
        ax.plot(lx, ly, color="steelblue", lw=1, label="Surviving adjacency (preview)")
        t0, t1 = native.time_qc.iloc[0], native.time_qc.iloc[-1]
        for method, times, lon, lat in self.references:
            mask = (times >= np.datetime64(t0)) & (times <= np.datetime64(t1))
            ok = position_valid(lon[mask], lat[mask])
            rx, ry = xy(lon[mask], lat[mask])
            ax.plot(np.where(ok, rx, np.nan), np.where(ok, ry, np.nan), "--", lw=1,
                    label=f"Supplied {method} (reference only)")
        event_edges = self.edges[self.edges.candidate_id == candidate.candidate_id]
        for number, edge in enumerate(event_edges.itertuples()):
            endpoints = self.frame.iloc[[int(edge.row_index_1), int(edge.row_index_2)]]
            ex, ey = xy(endpoints.lon_qc, endpoints.lat_qc)
            ax.plot(ex, ey, color="crimson", lw=2, alpha=.7,
                    label="Queue >3 m/s edges" if number == 0 else None)
        for decision, color, marker in [("keep", "forestgreen", "o"), ("reject", "crimson", "x"),
                                         ("uncertain", "darkorange", "D")]:
            chosen = native.manual_review_decision.eq(decision).to_numpy() & valid
            ax.scatter(x[chosen], y[chosen], c=color, marker=marker, s=60, label=decision, zorder=5)
        proposals = self.suggestions[self.suggestions.candidate_id == candidate.candidate_id]
        for confidence, color, marker in [("high", "purple", "s"), ("medium", "darkorange", "^")]:
            suggested_rows = proposals.loc[proposals.auto_confidence.eq(confidence), "row_index"].to_numpy(dtype=int)
            visible_rows = suggested_rows[(suggested_rows >= self.visible[0]) & (suggested_rows <= self.visible[-1])]
            offsets = visible_rows - self.visible[0]
            if len(offsets):
                ax.scatter(x[offsets], y[offsets], facecolors="none", edgecolors=color, marker=marker,
                           s=125, linewidths=1.8, label=f"{confidence} recovery proposal", zorder=4)
        stride = max(1, int(np.ceil(len(self.selectable) / 70)))
        for j in self.selectable[::stride]:
            offset = int(j - self.visible[0])
            ax.annotate(str(j), (x[offset], y[offset]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.scatter([0], [0], facecolors="none", edgecolors="black", s=180, lw=2, zorder=6, label="Selected")
        ax.set(xlabel="Local east (km)", ylabel="Local north (km)")
        # Preserve the map panel even for a straight east/west or north/south track.
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("Labels: chronological native row indices")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=.2)
        self.screen_rows = self.selectable.copy()
        sx, sy = xy(self.frame.lon_qc.iloc[self.screen_rows], self.frame.lat_qc.iloc[self.screen_rows])
        self.screen_xy = np.column_stack([sx, sy])
        window = (indices >= self.visible[0]) & (indices <= self.visible[-1])
        speed_ax.plot(self.frame.time.iloc[indices[window]], speeds[window], ".-", color="steelblue",
                      label="Surviving adjacency preview")
        for number, edge in enumerate(event_edges.itertuples()):
            speed_ax.scatter(self.frame.time.iloc[int(edge.row_index_2)], edge.speed_m_s, c="crimson", s=25,
                             label="Frozen queue flags" if number == 0 else None)
        speed_ax.axhline(3, color="crimson", ls=":", label="3 m/s")
        speed_ax.axvline(point.time, color="black", alpha=.6)
        speed_ax.set(xlabel="UTC time (speed assigned to later point)", ylabel="Edge speed (m/s)")
        speed_ax.tick_params(axis="x", labelrotation=25)
        speed_ax.legend(fontsize=7)
        speed_ax.grid(alpha=.2)
        speed_by_row = dict(zip(indices, speeds))
        later = indices[indices > self.selected]
        incoming = speed_by_row.get(self.selected, np.nan)
        outgoing = speed_by_row.get(int(later[0]), np.nan) if len(later) else np.nan
        self.figure.suptitle(f"{self.cursor + 1}/{len(self.queue)} | {candidate.platform_code} | "
                            f"iteration {self.state['review_iteration']} | {candidate.classification}\n"
                            f"{candidate.n_flagged_edges} flagged edges; max {candidate.max_speed_m_s:.2f} m/s", fontsize=12)
        self.info.set_text(f"Selected native row {self.selected}; source_obs_index {point.source_obs_index}; "
                           f"decision: {point.manual_review_decision or 'unreviewed'}\n"
                           f"UTC {utc_string(point.time)} | surviving incoming/outgoing: {incoming:.3f} / {outgoing:.3f} m/s\n"
                           "Click a point or use Left/Right. Edit reason/note, then K/R/U. Enter finishes text editing.")
        self.sync_text(candidate)
        counts = proposals.auto_confidence.value_counts()
        lines = [f"Frozen recovery proposals: high {counts.get('high', 0)}, medium {counts.get('medium', 0)}, "
                 f"unresolved {counts.get('unresolved', 0)}. A: accept eligible high blocks in this region."]
        selected_proposal = proposals[proposals.row_index == self.selected]
        if len(selected_proposal):
            proposal = selected_proposal.iloc[0]
            lines.append(f"Selected: {proposal.auto_confidence.upper()} | skip {proposal.auto_skip_count} | "
                         f"bridge {proposal.auto_bridge_speed_m_s:.3f} m/s | continuation "
                         f"{proposal.auto_continuation_speed_m_s:.3f} m/s | F/B agree: {proposal.auto_forward_backward_agree}")
            lines.append(f"Backward bridge / continuation: {proposal.auto_backward_bridge_speed_m_s:.3f} / "
                         f"{proposal.auto_backward_continuation_speed_m_s:.3f} m/s")
            for label, message in [("Forward", proposal.auto_forward_reason), ("Backward", proposal.auto_backward_reason),
                                    ("Reason", proposal.auto_reason)]:
                lines.extend(textwrap.wrap(f"{label}: {message.replace('_', ' ')}", width=145))
        else:
            lines.append("No skip proposal for the selected point. Manual decisions remain authoritative.")
        if "suggestions_sha256" not in self.state:
            lines = ["This queue predates recursive recovery. Close and run 'suggest' to generate proposals."]
        self.recovery_info.set_text("\n".join(lines))
        self.status.set_text("Decisions autosave. Queue stays fixed; close and run apply to rebuild it. Rejected points remain selectable.")
        self.figure.canvas.draw_idle()

    def save(self):
        self.decisions.save()
        write_json(self.session_path, {"review_iteration": self.state["review_iteration"], "cursor": self.cursor,
                                       "selected_row": int(self.selected) if self.selected is not None else None})

    def accept_high(self) -> int:
        """Explicit user action: revalidate whole blocks and never overwrite an override."""
        if self.state.get("recovery_configuration") != RECOVERY_CONFIG:
            raise ValueError("Recovery configuration changed or missing; run suggest before bulk acceptance")
        candidate = self.queue.iloc[self.cursor]
        frozen = self.suggestions[(self.suggestions.candidate_id == candidate.candidate_id)
                                  & self.suggestions.auto_suggested_reject]
        current = apply_decisions(self.native, self.decisions)
        live = recover_region(current, candidate, self.edges)
        live_high = live[live.auto_suggested_reject]
        accepted = []
        for block_id, block in frozen.groupby("auto_block_id"):
            original_rows = set(map(int, block.row_index))
            live_rows = set(map(int, live_high.loc[live_high.auto_block_id == block_id, "row_index"]))
            if original_rows != live_rows or not original_rows:
                continue
            if any(current.manual_review_decision.iloc[j] for j in original_rows):
                continue
            accepted.extend(block.to_dict("records"))
        previous = self.decisions.rows.copy()
        previous_hash = self.decisions.sha256
        try:
            for proposal in accepted:
                self.decisions.set(current.iloc[int(proposal["row_index"])], "reject",
                    iteration=int(self.state["review_iteration"]), candidate_id=candidate.candidate_id,
                    reason=f"user_accepted_high_confidence_recovery: {proposal['auto_reason']}",
                    note=f"proposal_block={proposal['auto_block_id']}; "
                         f"bridge={proposal['auto_bridge_speed_m_s']:.6g} m/s; "
                         f"continuation={proposal['auto_continuation_speed_m_s']:.6g} m/s")
            self.save()
        except (ValueError, OSError):
            # A cursor-file failure must not roll back an already persisted CSV.
            if self.decisions.sha256 == previous_hash:
                self.decisions.rows = previous
            raise
        return len(accepted)

    def action(self, key):
        try:
            if key in {"s", "q"}:
                self.save()
                self.status.set_text(f"Saved {len(self.decisions.rows)} explicit point decisions.")
                if key == "q":
                    self.plt.close(self.figure)
            elif not self.queue.empty:
                message = None
                if key == "a":
                    count = self.accept_high()
                    message = (f"Accepted {count} high-confidence suggested points. "
                               "Manual overrides and stale/incomplete blocks were preserved; run apply to regenerate.")
                elif key in {"k", "r", "u"}:
                    candidate = self.queue.iloc[self.cursor]
                    reason = self.reason_box.text.strip()
                    if not reason:
                        raise ValueError("Enter a reason before recording a decision")
                    self.decisions.set(self.frame.iloc[self.selected], {"k": "keep", "r": "reject", "u": "uncertain"}[key],
                                       iteration=int(self.state["review_iteration"]), reason=reason,
                                       candidate_id=candidate.candidate_id, note=self.note_box.text)
                    self.save()
                elif key in {"left", "right"}:
                    where = int(np.flatnonzero(self.selectable == self.selected)[0])
                    self.selected = int(self.selectable[np.clip(where + (1 if key == "right" else -1), 0, len(self.selectable) - 1)])
                elif key in {"n", "p"}:
                    self.cursor = int(np.clip(self.cursor + (1 if key == "n" else -1), 0, len(self.queue) - 1))
                    self.selected = None
                self.draw()
                if message is not None:
                    self.status.set_text(message)
        except (ValueError, OSError) as exc:
            self.status.set_text(f"NOT SAVED: {exc}")
        self.figure.canvas.draw_idle()

    def on_key(self, event):
        if not any(box.capturekeystrokes for box in (self.reason_box, self.note_box)):
            self.action((event.key or "").lower())

    def on_click(self, event):
        if event.inaxes != self.map_ax or self.queue.empty:
            return
        pixels = self.map_ax.transData.transform(self.screen_xy)
        distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        if len(distances) and distances.min() <= 15:
            self.selected = int(self.screen_rows[np.argmin(distances)])
            self.draw()

    def on_close(self, event):
        if self.closed:
            return
        try:
            self.save()
        except (ValueError, OSError) as exc:
            print(f"Could not save review on close: {exc}. Decisions successfully autosaved earlier remain on disk.")
        finally:
            self.closed = True
            self.master.close()

    def show(self):
        self.plt.show()
