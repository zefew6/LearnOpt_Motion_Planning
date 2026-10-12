"""Lazy compatibility exports for trajectory planning."""
from uav_ac.planning._lazy import install_exports

_EXPORTS = {'BandedPLU': ('uav_ac.planning.trajectory.gcopter.minco', 'BandedPLU'),
 'GCOPTER': ('uav_ac.planning.trajectory.gcopter.planner', 'GCOPTER'),
 'GCOPTERConfig': ('uav_ac.planning.trajectory.gcopter.config', 'GCOPTERConfig'),
 'GCOPTERTrajectory': ('uav_ac.planning.trajectory.gcopter.types', 'GCOPTERTrajectory'),
 'MINCOQuintic': ('uav_ac.planning.trajectory.gcopter.minco', 'MINCOQuintic'),
 'TrajectorySamples': ('uav_ac.planning.trajectory.gcopter.types', 'TrajectorySamples'),
 'evaluate_minco_objective': ('uav_ac.planning.trajectory.gcopter.optimization',
                              'evaluate_minco_objective'),
 'evaluate_minco_constraints': ('uav_ac.planning.trajectory.gcopter.optimization',
                                'evaluate_minco_constraints'),
 'DerivativeWaypoint': ('uav_ac.planning.trajectory.gcopter.optimization',
                        'DerivativeWaypoint'),
 'FixedWaypointMap': ('uav_ac.planning.trajectory.gcopter.optimization',
                      'FixedWaypointMap'),
 'FixedBoundaryDerivativeMap': ('uav_ac.planning.trajectory.gcopter.optimization',
                                'FixedBoundaryDerivativeMap')}

install_exports(globals(), _EXPORTS)
