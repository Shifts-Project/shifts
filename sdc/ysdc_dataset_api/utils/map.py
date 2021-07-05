import numpy as np

from ..proto.map_pb2 import MovementType, TrafficLightState


def repeated_points_to_array(geometry):
    poly = np.zeros((len(geometry.points), 2), dtype=np.float32)
    for i, point in enumerate(geometry.points):
        poly[i, 0] = point.x
        poly[i, 1] = point.y
    return poly


def is_traffic_light_rule_applicable(rule, traffic_light_sections):
    for rule_section in rule.sections:
        has_match = False
        for section in traffic_light_sections.sections:
            if rule_section.id == section.id and rule_section.state == section.state:
                has_match = True
                break
        if not has_match:
            return False
    return True


def get_crosswalk_availability(crosswalk, traffic_light_sections):
    if len(crosswalk.control_rules) == 0:
        # Unregulated crosswalk
        return 2.0
    else:
        for rule in crosswalk.control_rules:
            if (
                rule.movement_type != MovementType.MOVEMENT_FORBIDDEN and
                is_traffic_light_rule_applicable(rule, traffic_light_sections)
            ):
                return -1.0
    return 1.0


def get_lane_availability(lane, section_to_state):
    if not hasattr(lane, 'traffic_light_section_ids'):
        # TODO: Raise error
        return 1.
    else:
        if (
            not lane.traffic_light_section_ids.main_section_id
            and not lane.traffic_light_section_ids.left_section_id
            and not lane.traffic_light_section_ids.right_section_id
        ):
            # Unregulated lane
            return 2.
        elif (
            lane.traffic_light_section_ids.main_section_id
            and not lane.traffic_light_section_ids.left_section_id
            and not lane.traffic_light_section_ids.right_section_id
        ):
            section_id = lane.traffic_light_section_ids.main_section_id
            state = section_to_state.get(section_id, None)
            if (
                state is None
                or state == TrafficLightState.STATE_NOT_WORKING
                or state == TrafficLightState.STATE_DISABLED
            ):
                # Unregulated lane
                return 2.
            elif (
                state == TrafficLightState.STATE_RED
                or state == TrafficLightState.STATE_RED_YELLOW
                or state == TrafficLightState.STATE_YELLOW
            ):
                # Regulated disabled
                return -1.
            elif (
                state == TrafficLightState.STATE_GREEN
                or state == TrafficLightState.STATE_BLINKING_GREEN
                or state == TrafficLightState.STATE_BLINKING_RED
            ):
                # Regulated enabled
                return 1.
            elif (
                state == TrafficLightState.STATE_UNKNOWN
                or state == TrafficLightState.STATE_INVISIBLE
            ):
                # Regulated unknown
                return -2.
        elif (
            lane.traffic_light_section_ids.left_section_id
            or lane.traffic_light_section_ids.right_section_id
        ):
            main_state = section_to_state.get(lane.traffic_light_section_ids.main_section_id, None)
            additional_state = section_to_state.get(
                lane.traffic_light_section_ids.left_section_id
                or lane.traffic_light_section_ids.right_section_id,
                None
            )
            if (
                main_state == TrafficLightState.STATE_NOT_WORKING
                or main_state == TrafficLightState.STATE_DISABLED
            ):
                return 2.
            if (
                additional_state == TrafficLightState.STATE_UNKNOWN
                or additional_state == TrafficLightState.STATE_INVISIBLE
            ):
                return -2.
            elif additional_state == TrafficLightState.STATE_DISABLED:
                return -1.
            elif (
                additional_state == TrafficLightState.STATE_ENABLED
                or additional_state == TrafficLightState.STATE_BLINKING_ENABLED
            ):
                return 1.


def get_section_to_state(traffic_light_sections):
    res = {}
    for section in traffic_light_sections.sections:
        res[section.id] = section.state
    return res
