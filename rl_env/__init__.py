from rl_env.oloop1d import (
    OpenLoop1DTrack,
    OpenLoopStandard1DTrack,
    OpenLoopTeleportLong1DTrack
)

CUSTOM_ENVS = {
    "OLoopStandard1D": OpenLoopStandard1DTrack,
    "OLoopTeleportLong1D": OpenLoopTeleportLong1DTrack
}