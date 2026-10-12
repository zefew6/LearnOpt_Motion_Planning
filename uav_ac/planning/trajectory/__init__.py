"""Lazy compatibility exports for trajectory planning."""
from uav_ac.planning._lazy import install_exports

_EXPORTS = {'GCOPTER': ('uav_ac.planning.trajectory.gcopter', 'GCOPTER'),
 'GCOPTERConfig': ('uav_ac.planning.trajectory.gcopter', 'GCOPTERConfig'),
 'GCOPTERTrajectory': ('uav_ac.planning.trajectory.gcopter', 'GCOPTERTrajectory'),
 'GCSConfig': ('uav_ac.planning.trajectory.gcs', 'GCSConfig'),
 'GCSPlanner': ('uav_ac.planning.trajectory.gcs', 'GCSPlanner'),
 'GCSTrajectory': ('uav_ac.planning.trajectory.gcs', 'GCSTrajectory'),
 'MinimumSnap': ('uav_ac.planning.trajectory.minimum_snap', 'MinimumSnap'),
 'BMTPConfig': ('uav_ac.planning.trajectory.bmtp', 'BMTPConfig'),
 'BMTPLimits': ('uav_ac.planning.trajectory.bmtp', 'BMTPLimits'),
 'BMTPPlanner': ('uav_ac.planning.trajectory.bmtp', 'BMTPPlanner'),
 'BMTPResult': ('uav_ac.planning.trajectory.bmtp', 'BMTPResult'),
 'BMTPTrajectory': ('uav_ac.planning.trajectory.bmtp', 'BMTPTrajectory')}

install_exports(globals(), _EXPORTS)
