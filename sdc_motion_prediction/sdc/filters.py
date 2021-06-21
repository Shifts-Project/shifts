"""
** Filtering Info **

To filter scenes by tags one should specify a filter function
Scene tags dict has following structure:
{
    'day_time': one of {'kNight', 'kMorning', 'kAfternoon', 'kEvening'}
    'season': one of {'kWinter', 'kSpring', 'kSummer', 'kAutumn'}
    'track': one of {
      'Moscow' , 'Skolkovo', 'Innopolis', 'AnnArbor', 'Modiin', 'TelAviv'}
    'sun_phase': one of {'kAstronomicalNight', 'kTwilight', 'kDaylight'}
    'precipitation': one of {'kNoPrecipitation', 'kRain', 'kSleet', 'kSnow'}
}
Full description of protobuf message is available at
tags.proto file in sources

** Split Configuration **

Training Data ('train')
'moscow__train': Moscow intersected with NO precipitation

Validation Data ('validation')
'moscow__validation': Moscow intersected with NO precipitation
'ood__validation': Skolkovo, Modiin, and Innopolis intersected with
    (No precipitation, Rain and Snow)

Test Data ('test')
'moscow__test': Moscow intersected with NO precipitation
'ood__test': Ann-Arbor + Tel Aviv intersected with
    (No precipitation, rain, snow and sleet)
'moscow_precip__test': Moscow intersected with SLEET/SNOW/RAIN
"""


def filter_moscow_no_precipitation_data(scene_tags_dict):
    """
    This will need to be further divided into train/validation/test splits.
    """
    if (scene_tags_dict['track'] == 'Moscow' and
            scene_tags_dict['precipitation'] == 'kNoPrecipitation'):
        return True
    else:
        return False


def filter_moscow_precipitation_data(scene_tags_dict):
    if (scene_tags_dict['track'] == 'Moscow' and
            scene_tags_dict['precipitation'] != 'kNoPrecipitation'):
        return True
    else:
        return False


def filter_ood_validation_data(scene_tags_dict):
    if (scene_tags_dict['track'] in ['Skolkovo', 'Modiin', 'Innopolis'] and
        scene_tags_dict[
            'precipitation'] in ['kNoPrecipitation', 'kRain', 'kSnow']):
        return True
    else:
        return False


def filter_ood_test_data(scene_tags_dict):
    if scene_tags_dict['track'] in ['AnnArbor', 'TelAviv']:
        return True
    else:
        return False


DATASETS_TO_FILTERS = {
    'train': {
        'moscow__train': filter_moscow_no_precipitation_data
    },
    'validation': {
        'moscow__validation': filter_moscow_no_precipitation_data,
        'ood__validation': filter_ood_validation_data
    },
    'test': {
        'moscow__test': filter_moscow_no_precipitation_data,
        'ood__test': filter_ood_test_data,
        'moscow_precip__test': filter_moscow_precipitation_data
    }
}
