#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--cuda", default=True, action="store_true", help="use CUDA")
    parser.add_argument("--max_vali_f1", type=float, default=0)
    parser.add_argument("--config", type=str, default="experiments.conf")
    parser.add_argument("--mode", type=str, default="GD",
                        choices=["SGD", "GD", "SAG"])

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--h_feats', type=int, default=256,
                        help='hidden features')

    args = parser.parse_args()
    return args
