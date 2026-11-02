"""Parameter epochs: statistics that never pool sessions run under different configs.

Concept: an epoch is a maximal run of sessions with an identical config. Any
config change closes the current epoch and opens a new one. A session that was
in progress when the config changed is *interrupted*: it is recorded against the
epoch it ran under (as incomplete) and contributes to no statistic.

Pooling sessions across configs describes nothing (a mean over a mixture of
sigma = 2 and sigma = 5 is neither), so every aggregate lives on an Epoch and
is computed from that epoch's completed sessions only.

Paired comparison: every session is run for TWO strategies on identical mids,
fill draws and informed episodes (common random numbers): "naive" (eta = 0, no
detector, same delta) and "aware" (the configured strategy). Per session the
paired difference is  d_i = aware_i - naive_i  (terminal P&L); an epoch reports
mean(d), its standard error sd(d)/sqrt(n) and n. Because the strategies share
every random draw, sd(d) is much smaller than either strategy's own sd, which is
the point of pairing.

Sharpe here is the per-session ratio mean(terminal P&L) / sd(terminal P&L),
sample sd (ddof = 1), no annualisation and no risk-free rate.
Weakness: epochs with few sessions have wide standard errors; the table shows
n and se so a thin epoch is visibly thin. With n < 2 the se is undefined (nan).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

NAN = float("nan")


@dataclass(frozen=True)
class SessionRecord:
    """Outcome of one COMPLETED session (aware strategy; naive_* = paired naive run)."""

    index: int
    terminal_pnl: float
    spread_pnl: float
    inventory_pnl: float
    max_abs_q: float
    max_drawdown: float  # worst peak-to-trough of the intra-session MTM path, >= 0
    fills: float  # total filled SIZE (partial fills under the queue model)
    n_active: int  # ticks labelled informed = lead-in + active episode (scoring only)
    n_hit: int  # of those, ticks with detector alert
    n_quiet: int
    n_false: int
    clamp_steps: int = 0  # ticks where the min_dist quote floor bound
    naive_terminal_pnl: float = NAN
    naive_spread_pnl: float = NAN
    naive_inventory_pnl: float = NAN
    naive_max_abs_q: float = 0.0
    naive_max_drawdown: float = NAN


def _mean_sd_se(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    mean = float(x.mean()) if n else NAN
    sd = float(x.std(ddof=1)) if n > 1 else NAN
    return mean, sd, sd / np.sqrt(n) if n > 1 else NAN


@dataclass
class Epoch:
    id: int
    config: dict
    sessions: list = field(default_factory=list)  # SessionRecord
    interrupted: list = field(default_factory=list)  # indices of incomplete sessions
    start_session: int | None = None  # index of the first session run under this config

    def paired_diffs(self) -> np.ndarray:
        """d_i = aware_i - naive_i, terminal P&L, one per completed session."""
        return np.array([s.terminal_pnl - s.naive_terminal_pnl for s in self.sessions],
                        dtype=float)

    def stats(self) -> dict:
        pnl = np.array([s.terminal_pnl for s in self.sessions], dtype=float)
        npnl = np.array([s.naive_terminal_pnl for s in self.sessions], dtype=float)
        n = len(pnl)
        mean, sd, se = _mean_sd_se(pnl)
        n_mean, n_sd, n_se = _mean_sd_se(npnl)
        d_mean, d_sd, d_se = _mean_sd_se(self.paired_diffs())
        act = sum(s.n_active for s in self.sessions)
        quiet = sum(s.n_quiet for s in self.sessions)
        return dict(
            n=n,
            incomplete=len(self.interrupted),
            mean=mean,
            sd=sd,
            se=se,
            sharpe=mean / sd if n > 1 and sd > 0 else NAN,
            naive_mean=n_mean,
            naive_sd=n_sd,
            naive_se=n_se,
            naive_sharpe=n_mean / n_sd if n > 1 and n_sd > 0 else NAN,
            diff_mean=d_mean,
            diff_sd=d_sd,
            diff_se=d_se,
            diff_t=d_mean / d_se if n > 1 and d_se > 0 else NAN,
            worst_dd=max((s.max_drawdown for s in self.sessions), default=NAN),
            naive_worst_dd=max((s.naive_max_drawdown for s in self.sessions), default=NAN),
            max_abs_q=max((s.max_abs_q for s in self.sessions), default=0),
            naive_max_abs_q=max((s.naive_max_abs_q for s in self.sessions), default=0),
            detect=sum(s.n_hit for s in self.sessions) / act if act else NAN,
            false_alarm=sum(s.n_false for s in self.sessions) / quiet if quiet else NAN,
        )

    def running_paired(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Running mean and se of d after 1, 2, ... sessions (se is nan at 1)."""
        d = self.paired_diffs()
        k = np.arange(1, len(d) + 1, dtype=float)
        mean = np.cumsum(d) / k if len(d) else d
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (np.cumsum(d**2) - k * mean**2) / (k - 1)
            se = np.sqrt(np.maximum(var, 0.0) / k)
        if len(d):
            se[0] = NAN
        return k, mean, se


def config_str(cfg: Mapping, base: Mapping | None = None) -> str:
    """'k=v ...'; with `base`, only the keys that differ ('k a->b')."""
    parts = []
    for k, v in cfg.items():
        if base is None:
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
        elif base.get(k) != v:
            old = base.get(k)
            f = lambda x: f"{x:g}" if isinstance(x, float) else str(x)
            parts.append(f"{k} {f(old)}->{f(v)}")
    return " ".join(parts)


class EpochLog:
    def __init__(self, config: Mapping, start_session: int = 0) -> None:
        self.epochs: list[Epoch] = [Epoch(0, dict(config), start_session=start_session)]

    @property
    def current(self) -> Epoch:
        return self.epochs[-1]

    def update_config(self, config: Mapping, interrupted: int | None = None,
                      start_session: int | None = None) -> bool:
        """Open a new epoch if `config` differs from the current one.

        `interrupted` = index of the session that was mid-flight; it is recorded
        as incomplete on the epoch it ran under (the one being closed).
        `start_session` = index of the first session run under the new config.
        Returns True if a new epoch was opened.
        """
        if dict(config) == self.current.config:
            return False
        if interrupted is not None:
            self.current.interrupted.append(interrupted)
        self.epochs.append(Epoch(len(self.epochs), dict(config), start_session=start_session))
        return True

    def add_session(self, rec: SessionRecord) -> None:
        self.current.sessions.append(rec)

    @property
    def n_completed(self) -> int:
        return sum(len(e.sessions) for e in self.epochs)

    @property
    def n_incomplete(self) -> int:
        return sum(len(e.interrupted) for e in self.epochs)

    def history_lines(self, last: int = 4) -> list[str]:
        """Compact lines for CLOSED epochs that completed at least one session."""
        done = [e for e in self.epochs[:-1] if e.sessions]
        lines = []
        for e in done[-last:]:
            s = e.stats()
            lines.append(f"#{e.id}: n={s['n']:<3d} diff {s['diff_mean']:+7.2f}±{s['diff_se']:.2f}")
        return lines

    def table(self) -> str:
        shown = [e for e in self.epochs if e.sessions or e.interrupted]
        empty = len(self.epochs) - len(shown)
        head = (f"{'ep':>3} {'n':>5} {'incmp':>5} | {'aware mean':>10} {'se':>7} {'Sharpe':>7} | "
                f"{'naive mean':>10} {'Sharpe':>7} | {'diff mean':>9} {'se':>7} {'t':>6} | "
                f"{'worstDD a/n':>12} {'max|q| a/n':>10} {'detect':>7} {'false':>7}  "
                f"config (diff vs previous)")
        rows = [head, "-" * len(head)]
        prev = None
        for e in shown:
            s = e.stats()
            cfg = config_str(e.config) if prev is None else (config_str(e.config, prev) or "(same)")
            rows.append(
                f"{e.id:>3d} {s['n']:>5d} {s['incomplete']:>5d} | {s['mean']:>10.2f} {s['se']:>7.2f} "
                f"{s['sharpe']:>7.2f} | {s['naive_mean']:>10.2f} {s['naive_sharpe']:>7.2f} | "
                f"{s['diff_mean']:>9.2f} {s['diff_se']:>7.2f} {s['diff_t']:>6.2f} | "
                f"{s['worst_dd']:>5.1f}/{s['naive_worst_dd']:<6.1f} "
                f"{s['max_abs_q']:>4.1f}/{s['naive_max_abs_q']:<5.1f} "
                f"{s['detect']:>7.1%} {s['false_alarm']:>7.1%}  {cfg}")
            prev = e.config
        if empty:
            rows.append(f"({empty} epoch(s) with no sessions at all omitted)")
        return "\n".join(rows)
