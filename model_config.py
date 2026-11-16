"""Read model architecture settings without adding a configuration dependency."""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'base.json'


def _require_keys(value, expected, section):
    if not isinstance(value, Mapping):
        raise ValueError(f'{section} must be a JSON object.')
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ValueError(f'{section}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}.')


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def _blocks(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a nonempty list of block counts.')
    for count in value:
        _integer(count, name, minimum=0)


def load_model_config(source=None):
    """Load a complete JSON config, a saved config dict, or the default Base."""
    if isinstance(source, Mapping):
        config = deepcopy(dict(source))
    else:
        path = DEFAULT_CONFIG if source is None else Path(source).expanduser()
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot read model config {path}: {error}') from error
    _require_keys(config, {'name', 'restoration', 'enhancement'}, 'model config')
    if not isinstance(config['name'], str) or not config['name'].strip():
        raise ValueError('name must be a nonempty string.')
    restoration, enhancement = config['restoration'], config['enhancement']
    _require_keys(restoration, {'img_channel', 'width', 'middle_blk_num',
                                'enc_blk_nums', 'dec_blk_nums'}, 'restoration')
    _require_keys(enhancement, {'ci', 'grid_range', 'scale_factor', 'co', 'kernel',
                                'layer', 'width', 'enc_blks', 'dec_blks',
                                'middle_blk_num'}, 'enhancement')
    for section_name, section in (('restoration', restoration), ('enhancement', enhancement)):
        _integer(section['width'], f'{section_name}.width')
        _integer(section['middle_blk_num'], f'{section_name}.middle_blk_num', minimum=0)
    for section, enc, dec in ((restoration, 'enc_blk_nums', 'dec_blk_nums'),
                              (enhancement, 'enc_blks', 'dec_blks')):
        _blocks(section[enc], enc)
        _blocks(section[dec], dec)
        if len(section[enc]) != len(section[dec]):
            raise ValueError(f'{enc} and {dec} must contain the same number of stages.')
    _integer(restoration['img_channel'], 'restoration.img_channel')
    for key in ('ci', 'grid_range', 'co', 'kernel', 'layer'):
        _integer(enhancement[key], f'enhancement.{key}')
    if restoration['img_channel'] != 4 or enhancement['ci'] != 3 or enhancement['co'] != 4:
        raise ValueError('The packed-RAW/RGB pipeline requires img_channel=4, ci=3, and co=4.')
    if enhancement['kernel'] % 2 == 0 or enhancement['layer'] < 2:
        raise ValueError('The guide network requires an odd kernel and layer >= 2.')
    scale = enhancement['scale_factor']
    if type(scale) not in (int, float) or not 0 < scale <= 1:
        raise ValueError('enhancement.scale_factor must be a number in (0, 1].')
    return config


def resolve_model_config(source=None, checkpoint=None):
    """Prefer an explicit config, then checkpoint metadata, then Base.

    Config metadata also captures settings such as scale_factor that cannot be
    inferred from parameter shapes in a state dict.
    """
    explicit = load_model_config(source) if source is not None else None
    saved = checkpoint.get('model_config') if checkpoint is not None else None
    saved = load_model_config(saved) if saved is not None else None
    if explicit is not None and saved is not None:
        if any(explicit[key] != saved[key] for key in ('restoration', 'enhancement')):
            raise ValueError('Model config does not match the checkpoint. Select the matching '
                             '--config or omit --config to use its saved model configuration.')
    if explicit is not None:
        return explicit
    if saved is not None:
        return saved
    return load_model_config()
