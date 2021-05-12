import numpy as np

from ..proto.map_pb2 import MovementType


def get_polygon(geometry):
    poly = np.zeros((len(geometry.points), 2))
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
