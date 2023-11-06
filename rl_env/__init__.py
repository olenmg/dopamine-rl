from rl_env.oloop1d import (
    OpenLoop1DTrack,
    OpenLoopStandard1DTrack,
    OpenLoopTeleportLong1DTrack
)

from rl_env.ball_crashing import BallCrashing

CUSTOM_ENVS = {
    "OLoopStandard1D": OpenLoopStandard1DTrack,
    "OLoopTeleportLong1D": OpenLoopTeleportLong1DTrack,
    "BallCrashing": BallCrashing
}