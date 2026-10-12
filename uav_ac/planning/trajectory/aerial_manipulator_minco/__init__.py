"""Lazy compatibility exports for trajectory planning."""
from uav_ac.planning._lazy import install_exports

_EXPORTS = {'AerialManipulatorMINCO': ('uav_ac.planning.trajectory.aerial_manipulator_minco.planner',
                            'AerialManipulatorMINCO'),
 'AerialManipulatorMINCOConfig': ('uav_ac.planning.trajectory.aerial_manipulator_minco.config',
                                  'AerialManipulatorMINCOConfig'),
 'AerialManipulatorSearchResult': ('uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory',
                                   'AerialManipulatorSearchResult'),
 'AerialManipulatorTrajectory': ('uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory',
                                 'AerialManipulatorTrajectory'),
 'make_terminal_state': ('uav_ac.planning.trajectory.aerial_manipulator_minco.search',
                         'make_terminal_state'),
 'JointTaskTrajectory': ('uav_ac.planning.trajectory.aerial_manipulator_minco.trajectory',
                         'JointTaskTrajectory'),
 'TaskWaypoint': ('uav_ac.planning.trajectory.aerial_manipulator_minco.constraints',
                  'TaskWaypoint'),
 'TaskEventConstraint': ('uav_ac.planning.trajectory.aerial_manipulator_minco.constraints',
                         'TaskEventConstraint')}

install_exports(globals(), _EXPORTS)
