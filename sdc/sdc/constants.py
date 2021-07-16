VALID_TRAJECTORY_TAGS = [
    'kMoveLeft', 'kMoveRight', 'kMoveForward', 'kMoveBack',
    'kAcceleration', 'kDeceleration', 'kUniform',
    'kStopping', 'kStarting', 'kStationary']
SCENE_TAG_TYPE_TO_OPTIONS = {
    'day_time': ['kNight', 'kMorning', 'kAfternoon', 'kEvening'],
    'season': ['kWinter', 'kSpring', 'kSummer', 'kAutumn'],
    'track': [
        'Moscow', 'Skolkovo', 'Innopolis', 'AnnArbor', 'Modiin', 'TelAviv'],
    'sun_phase': ['kAstronomicalNight', 'kTwilight', 'kDaylight'],
    'precipitation': ['kNoPrecipitation', 'kRain', 'kSleet', 'kSnow']
}
VALID_AGGREGATORS = {'min', 'avg', 'top1', 'weighted'}
VALID_BASE_METRICS = {'ade', 'fde'}
SPLIT_TO_PB_DATASET_PATH = {  # Protobufs (unrendered)
    'train': '/train_pb/',
    'validation': '/validation_pb/',
}
SPLIT_TO_RENDERED_DATASET_PATH = {  # np.ndarrays (rendered)
    'train': '/train_rendered/',
    'validation': '/validation_rendered/',
}
SPLIT_TO_SCENE_TAGS_PATH = {
    'train': '/train_tags.txt',
    'validation': '/validation_tags.txt',
}

# Matplotlib colors for retention plots with baselines.
# Matches paper color scheme.
BASELINE_TO_COLOR_HEX = {
    'Optimal': '#2CA02C',  # Green
    'Baseline': '#FF7F0E',  # Orange
    'Random': '#1F77B4'  # Blue
}

RENDERER_CONFIG = {
    # parameters of feature maps to render
    'feature_map_params': {
        'rows': 400,
        'cols': 400,
        'resolution': 0.25,  # number of meters in one pixel
    },
    'renderers_groups': [
        # Having several feature map groups
        # allows to independently render feature maps with different history length.
        # This could be useful to render static features (road graph, etc.) once.
        {
            # start: int, first timestamp into the past to render, 0 – prediction time
            # stop: int, last timestamp to render inclusively, 24 – farthest known point into the past
            # step: int, grid step size,
            #            step=1 renders all points between start and stop,
            #            step=2 renders every second point, etc.
            'time_grid_params': {
                'start': 0,
                'stop': 0,
                'step': 1,
            },
            'renderers': [
                # each value is rendered at its own channel
                # occupancy -- 1 channel
                # velocity -- 2 channels (x, y)
                # acceleration -- 2 channels (x, y)
                # yaw -- 1 channel
                {'vehicles': ['occupancy', 'velocity', 'acceleration', 'yaw']},
                # only occupancy and velocity are available for pedestrians
                {'pedestrians': ['occupancy', 'velocity']},
            ]
        },
        {
            'time_grid_params': {
                'start': 0,
                'stop': 0,
                'step': 1,
            },
            'renderers': [
                {
                    'road_graph': [
                        'crosswalk_occupancy',
                        'crosswalk_availability',
                        'lane_availability',
                        'lane_direction',
                        'lane_occupancy',
                        'lane_priority',
                        'lane_speed_limit',
                        'road_polygons',
                    ]
                }
            ]
        }
    ]
}
