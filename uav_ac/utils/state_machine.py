"""Reusable state machine with caller-defined states and transitions."""

from collections.abc import Hashable, Mapping, Set
from typing import Generic, TypeVar


StateT = TypeVar("StateT", bound=Hashable)


class StateMachine(Generic[StateT]):
    """Track a state and its history while enforcing a caller-supplied graph."""

    def __init__(self, initial_state: StateT,
                 transitions: Mapping[StateT, Set[StateT]]) -> None:
        self._allowed = {
            state: frozenset(targets)
            for state, targets in transitions.items()
        }
        if initial_state not in self._allowed:
            raise ValueError("initial state must be present in the transition map")
        unknown_targets = {
            target for targets in self._allowed.values() for target in targets
            if target not in self._allowed
        }
        if unknown_targets:
            raise ValueError(f"transition targets are missing from the map: {unknown_targets!r}")

        self.state = initial_state
        self.history = [initial_state]
        self.last_reason: str | None = None

    def transition(self, target: StateT, reason: str | None = None) -> None:
        if target not in self._allowed[self.state]:
            source_name = getattr(self.state, "name", repr(self.state))
            target_name = getattr(target, "name", repr(target))
            raise ValueError(f"invalid state transition {source_name} -> {target_name}")
        self.state = target
        self.history.append(target)
        self.last_reason = reason
