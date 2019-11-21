"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class DemoOptions(BaseOptions):
    def initialize(self, parser):
        self.isTrain = False
        BaseOptions.initialize(self, parser)
        parser.set_defaults(netG='spadeladder', loadSize=256, use_vae=True, which_epoch='latest', name=None,
                            continue_train=True)
        parser.set_defaults(dataset_mode='safebooru')
        parser.set_defaults(checkpoints_dir='../checkpoints')
        parser.add_argument('-s', '--simplification', type=str,
                            default='../utils/preprocessing/sketch_simplification/model.pth',
                            help='Directory of sketch simplification model')
        parser.add_argument('--port', type=int, default=41234, help="Server port number")
        parser.add_argument('--cuda', action='store_true')
        return parser
