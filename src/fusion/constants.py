from __future__ import annotations

DETECTION_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CATEGORY_TO_DETECTION = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

RADAR_CHANNELS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

DEFAULT_ATTRIBUTES = {
    "car": "vehicle.parked",
    "truck": "vehicle.parked",
    "construction_vehicle": "vehicle.parked",
    "bus": "vehicle.stopped",
    "trailer": "vehicle.parked",
    "barrier": "",
    "motorcycle": "cycle.without_rider",
    "bicycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
    "traffic_cone": "",
}
