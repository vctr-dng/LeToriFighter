"""Success detectors on synthetic observations (no game)."""

import numpy as np
import pytest

from letorifighter.tasks import (
    CONTACT_THRESHOLD,
    ControlDetector,
    EscapeDetector,
    ReachDetector,
    SequenceDetector,
    geometry,
)
from letorifighter.testing import (
    START_DISTANCE,
    make_observation,
    obs_at_distance,
    obs_with_hand_contact,
)

FAR = obs_at_distance(START_DISTANCE)


def feed(detector, observations):
    return [detector.update(obs) for obs in observations]


def test_start_pose_distance_and_no_contact():
    assert geometry.groin_distance(FAR) == pytest.approx(START_DISTANCE, abs=1e-5)
    assert not any(geometry.hand_contacts(FAR).values())
    assert geometry.reach_margin(FAR) > 0


@pytest.mark.parametrize("target", list(CONTACT_THRESHOLD))
def test_contact_threshold_edges(target):
    limit = CONTACT_THRESHOLD[target]
    inside = obs_with_hand_contact("l_hand", target, gap=limit - 0.02)
    outside = obs_with_hand_contact("l_hand", target, gap=limit + 0.02)
    only = (target,)  # neighbouring limbs are closer than the threshold
    assert geometry.hand_contacts(inside, only)["l_hand"]
    assert not geometry.hand_contacts(outside, only)["l_hand"]
    assert geometry.reach_margin(inside, only) < 0
    assert geometry.reach_margin(outside, only) > 0


def test_reach_needs_two_consecutive_contact_turns():
    touch = obs_with_hand_contact("r_hand", "chest", gap=0.05)
    detector = ReachDetector()
    statuses = feed(detector, [FAR, touch, FAR, touch, touch])
    assert [s.success for s in statuses] == [False, False, False, False, True]
    assert statuses[3].streak == 1 and statuses[4].streak == 2


def test_success_is_latched():
    touch = obs_with_hand_contact()
    detector = ReachDetector(hold_turns=1)
    statuses = feed(detector, [touch, FAR, FAR])
    assert [s.success for s in statuses] == [True, True, True]
    detector.reset()
    assert not detector.update(FAR).success


def test_control_keep_close_range_and_hold():
    detector = ControlDetector("keep_close", hold_turns=3)
    close = obs_at_distance(0.7)
    statuses = feed(detector, [close, close, obs_at_distance(1.3), close, close, close])
    assert [s.success for s in statuses] == [False] * 5 + [True]
    assert not ControlDetector("keep_close").condition(obs_at_distance(0.2))
    assert not ControlDetector("keep_close").condition(obs_at_distance(1.01))


def test_control_keep_distance_band():
    detector = ControlDetector("keep_distance")
    assert detector.condition(obs_at_distance(1.8))
    assert not detector.condition(obs_at_distance(1.3))
    assert not detector.condition(obs_at_distance(2.6))


def test_control_clinch_needs_contact_and_matching_grip():
    detector = ControlDetector("clinch", hold_turns=2)
    no_grip = obs_with_hand_contact("r_hand", "chest", gap=0.05, grips=(0.0, 0.0))
    wrong_hand = obs_with_hand_contact("r_hand", "chest", gap=0.05, grips=(1.0, 0.0))
    right = obs_with_hand_contact("r_hand", "chest", gap=0.05, grips=(0.0, 1.0))
    left = obs_with_hand_contact("l_hand", "head", gap=0.05, grips=(1.0, 0.0))
    assert not detector.condition(no_grip)
    assert not detector.condition(wrong_hand)
    assert detector.condition(right) and detector.condition(left)
    assert feed(detector, [right, left])[-1].success


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        ControlDetector("nope")


def test_escape_distance_hold_and_height():
    detector = EscapeDetector()
    far = obs_at_distance(2.8)
    statuses = feed(detector, [far, far, obs_at_distance(1.0), far, far, far])
    assert [s.success for s in statuses] == [False] * 5 + [True]
    low = obs_at_distance(2.8)
    low["self"]["groin_height"] = np.array([0.1], np.float32)
    assert EscapeDetector().condition(low)
    assert not EscapeDetector(min_groin_height=0.5).condition(low)


def test_sequence_stages_run_in_order():
    touch = obs_with_hand_contact("r_hand", "head", gap=0.05, distance=0.9)
    detector = SequenceDetector(
        [ReachDetector(hold_turns=1), ControlDetector("keep_close", hold_turns=2)]
    )
    # Reach is satisfied by the first observation, but the second stage only
    # starts counting from the next one.
    close = obs_at_distance(0.8)
    statuses = feed(detector, [touch, close, close])
    assert [s.success for s in statuses] == [False, False, True]
    assert [s.stage for s in statuses] == [0, 1, 1]


def test_sequence_does_not_succeed_if_second_stage_came_first():
    detector = SequenceDetector(
        [ReachDetector(hold_turns=1), ControlDetector("keep_close", hold_turns=1)]
    )
    close = obs_at_distance(0.8)
    assert not feed(detector, [close, close])[-1].success


def test_make_observation_defaults_are_a_valid_observation():
    obs = make_observation()
    assert obs["self"]["positions"].shape == (21, 3)
