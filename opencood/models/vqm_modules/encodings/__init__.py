from .pillars import Pillars
from .second import SECOND
from .liftsplat import LiftSplat
from .unproj import Unprojection
from .modal_fusion import ModalFusionBlock


def build_encoder(args, device):
    assert 'method' in args, '`method` should be in args.'
    if args['method'] == 'pillar':
        return Pillars(args, device)
    if args['method'] == 'second':
        return SECOND(args, device)
    if args['method'] == 'liftsplat':
        return LiftSplat(args, device)
    if args['method'] == 'unproj':
        return Unprojection(args, device)
    print(f'methods for agent must be [pillar|second|liftsplat|unproj].')
    raise
