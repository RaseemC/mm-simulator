"""Parameter dataclasses with defaults and input validation.

Concept: every tunable number lives here, so a run is fully described by a
config object that can be printed at the top of any script's output.

Defaults are the Avellaneda-Stoikov (2008) baseline:
    m_0 = 100, T = 1, N = 200 (dt = T/N = 0.005), sigma = 2,
    A = 140, k = 1.5, delta = 0.75, eta = 0 (naive) vs 0.1 (aware).

Validation rules (raise ValueError):
    * sigma >= 0, T > 0, N >= 1, A >= 0, k >= 0, delta >= 0, eta >= 0
    * A * dt <= 1, otherwise the per-step fill probability A*exp(-k d)*dt could
      exceed 1 even at zero distance and the min(1, .) cap would always bind.

Adverse selection (AdverseConfig): hidden "informed episodes" with a direction
s = +-1. During an episode the mid drifts by  strength * sigma * sqrt(dt) * s
per tick (strength is in units of the diffusion scale, so it means the same
thing at any sigma) and the side about to be wrong is hit `fill_boost` times as
often. strength = 0 switches the whole mechanism off.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketConfig:
    """Midprice process parameters: m_{n+1} = m_n + sigma*sqrt(dt)*Z."""

    m0: float = 100.0
    T: float = 1.0
    N: int = 200
    sigma: float = 2.0

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if int(self.N) != self.N or self.N < 1:
            raise ValueError(f"N must be a positive integer, got {self.N}")

    @property
    def dt(self) -> float:
        return self.T / self.N


@dataclass(frozen=True)
class FillConfig:
    """Fill intensity: p = min(1, A * exp(-k * distance) * dt).

    min_dist is a floor on every quote distance, applied inside core.step so live
    and batch share it: d = max(d_raw, min_dist). It stops strong inventory skew
    from putting a quote on the wrong side of the mid. -inf disables the floor.
    """

    A: float = 140.0
    k: float = 1.5
    min_dist: float = 0.02

    def __post_init__(self) -> None:
        if self.A < 0:
            raise ValueError(f"A must be >= 0, got {self.A}")
        if self.k < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")


@dataclass(frozen=True)
class AdverseConfig:
    """Hidden informed-flow episodes (see module docstring).

    Episode lifecycle per session: while quiet, each tick starts an episode with
    probability p_episode. It has `lead` ticks in which only the order book tilts
    (the synthetic book is generated from the hidden direction), then lasts
    1 + floor(Exp(mean_duration)) active ticks with drift and boosted fills.

    Default strength 0.12 is calibrated (scripts/calibrate_toxicity.py) so that
    informed episodes add about 15% to the variance of a session's net move
    (E[(m_N - m_0)^2], paired against strength 0). The realised variance
    (sum of squared tick changes) rises only ~0.2% at that strength: it is nearly
    blind to drift. Matching +10-20% in THAT sense would need strength ~ 1, which
    inflates the net-move variance ~11x.
    """

    strength: float = 0.12      # drift per active tick, in units of sigma*sqrt(dt); see below
    p_episode: float = 0.004
    lead: int = 10
    mean_duration: float = 60.0
    fill_boost: float = 2.5     # fill-probability multiplier on the picked-off side

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError(f"strength must be >= 0, got {self.strength}")
        if not 0 <= self.p_episode <= 1:
            raise ValueError(f"p_episode must be in [0, 1], got {self.p_episode}")
        if int(self.lead) != self.lead or self.lead < 1:
            raise ValueError(f"lead must be an integer >= 1, got {self.lead}")
        if self.mean_duration <= 0:
            raise ValueError(f"mean_duration must be > 0, got {self.mean_duration}")
        if self.fill_boost < 1:
            raise ValueError(f"fill_boost must be >= 1, got {self.fill_boost}")


@dataclass(frozen=True)
class StrategyConfig:
    """Quoting parameters: half-spread delta and inventory skew eta."""

    delta: float = 0.75
    eta: float = 0.0  # 0 = naive symmetric quoting

    def __post_init__(self) -> None:
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if self.eta < 0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")


@dataclass(frozen=True)
class QueueConfig:
    """Limit-order-book / queue-position fill model (see book.py).

    Prices live on an absolute grid of size `tick`. `levels` price levels per side
    are modelled, measured outward from the mid's grid anchor, so the window spans
    `levels * tick` (default 4.0) each way. A quote beyond it rests but cannot be
    filled; the default covers delta + eta*|q| over the eta range we sweep. Sizes are continuous
    "lots" (a deliberate simplification: it exercises partial fills without an
    integer-lot model), and `our_size` is the size we post on each side.

    Flow per tick, per side:
      market orders  arrive with probability `mo_rate` and size Exp(`mo_size_mean`),
                     consuming depth from the touch outward;
      cancellations  remove a fraction Uniform(0, 2*`cancel_rate`) of resting size,
                     which advances anyone behind them in the queue;
      adds           join the BACK of each level, size `add_rate` * target depth;
      replenishment  pulls depth back toward its target profile at `replenish_rate`.

    Target depth profile at level j (0 = touch):
        target[j] = depth_base * (1 + depth_slope * j) * exp(+-tilt) * M_j
    where M_j is a per-session lognormal multiplier of log-sd `depth_cv` and the
    tilt is the hidden informed direction (bid side +, ask side -), as before.

    During an ACTIVE informed episode the market-order rate on the side that picks
    us off is multiplied by `informed_mo_mult`; that is this model's adverse
    selection, replacing the exponential model's `fill_boost`.

    Defaults here are PROVISIONAL, pending calibration against published order-book
    statistics (B+3); they are not fitted to any instrument. They were chosen so the
    model trades at a rate comparable to the exponential model it replaces: at the
    default half-spread (delta = 0.75 = 3 ticks) a resting order is filled only when
    a market order clears every level in front of it, which needs
        size > cumulative depth through level 2 + our level's depth ~ 49 lots,
    so fills/tick/side ~ mo_rate * exp(-49 / mo_size_mean) ~ 0.17, against 0.227 for
    p = A*exp(-k*0.75)*dt. With the first guesses (depth_base 20, mo_size_mean 12)
    that threshold was 122 lots and the model never traded at all - see DESIGN_NOTES.
    """

    tick: float = 0.25
    levels: int = 16
    depth_base: float = 8.0
    depth_slope: float = 0.35
    depth_cv: float = 0.30
    mo_rate: float = 0.50
    mo_size_mean: float = 45.0
    cancel_rate: float = 0.08
    add_rate: float = 0.06
    replenish_rate: float = 0.15
    our_size: float = 1.0
    tilt: float = 0.7
    informed_mo_mult: float = 2.5
    keep_priority: bool = False
    """Counterfactual switch, NOT a market feature.

    False (default, realistic): moving our quote cancels the old order and rejoins
    at the back of the new level's queue. True: the order keeps its place in the
    queue even when its price moves, which no real venue allows. Running a pair of
    otherwise identical simulations with this False and True isolates the cost of
    requoting, since every other draw is shared.
    """

    def __post_init__(self) -> None:
        if self.tick <= 0:
            raise ValueError(f"tick must be > 0, got {self.tick}")
        if int(self.levels) != self.levels or not 1 <= self.levels <= 24:
            raise ValueError(f"levels must be an integer in 1..24, got {self.levels}")
        for name in ("depth_base", "mo_size_mean", "our_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("depth_slope", "depth_cv", "add_rate", "tilt"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("mo_rate", "cancel_rate", "replenish_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.informed_mo_mult < 1:
            raise ValueError(f"informed_mo_mult must be >= 1, got {self.informed_mo_mult}")


@dataclass(frozen=True)
class SimConfig:
    """Everything needed to describe one simulation run."""

    market: MarketConfig = field(default_factory=MarketConfig)
    fills: FillConfig = field(default_factory=FillConfig)
    adverse: AdverseConfig = field(default_factory=AdverseConfig)
    naive: StrategyConfig = field(default_factory=lambda: StrategyConfig(eta=0.0))
    aware: StrategyConfig = field(default_factory=lambda: StrategyConfig(eta=0.1))
    seed: int = 12345
    n_sessions: int = 10_000
    liquidation_penalty: float = 0.0  # c in: terminal P&L = Pi_N - c*|q_N|
    reset_mid: bool = True  # live mode: new session restarts at m_0 (else continues)

    def __post_init__(self) -> None:
        if self.n_sessions < 1:
            raise ValueError("n_sessions must be >= 1")
        if self.liquidation_penalty < 0:
            raise ValueError("liquidation_penalty must be >= 0")
        if self.fills.A * self.market.dt > 1:
            raise ValueError(
                f"A*dt = {self.fills.A * self.market.dt:.3f} > 1: fill probability "
                "would exceed 1 even at zero distance. Increase N or reduce A."
            )
