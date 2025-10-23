import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import (
    PoolingLayer, ResidualBlock, ResidualBlock3x3, ResidualBlock5x5, ResidualBlock7x7, SpatialSE, ChannelSE, 
    ResidualBlockDepthwise3x3, ResidualBlockDepthwise5x5, ResidualBlockDepthwise7x7, ResidualBlockDepthwise9x9, 
    DummyBlock, Conv3x3PoolingLayer, Depthwise3x3ConvPoolingLayer, MaxPoolingLayer, AvgPoolingLayer,
    Conv5x5PoolingLayer, Depthwise5x5ConvPoolingLayer, Depthwise7x7ConvPoolingLayer, AdaptiveRouter,
    ConvStem, DepthwiseConvStem, SpatialAttention, AdaptiveRouterMultiStep, SparseAdaptiveRouterMultiStep
)

def get_init_stem(filters, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=[
            ConvStem(filters, 3),
            ConvStem(filters, 5),
            ConvStem(filters, 7),
            DepthwiseConvStem(filters, 3),
            DepthwiseConvStem(filters, 5),
            DepthwiseConvStem(filters, 7),
            DepthwiseConvStem(filters, 9),
        ],
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )


def get_init_pooling(filters, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 64,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=[
            Conv3x3PoolingLayer(filters),
            Conv5x5PoolingLayer(filters, groups=2),
            Depthwise3x3ConvPoolingLayer(filters),
            Depthwise5x5ConvPoolingLayer(filters),
            Depthwise7x7ConvPoolingLayer(filters),
            MaxPoolingLayer(filters),
            AvgPoolingLayer(filters)
        ],
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )


def get_init_block(filters, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=[
            ResidualBlock3x3(filters),
            ResidualBlock5x5(filters, groups=2),
            ResidualBlockDepthwise3x3(filters),
            ResidualBlockDepthwise5x5(filters),
            ResidualBlockDepthwise7x7(filters),
            ChannelSE(ratio=4),
            SpatialSE(),
            SpatialAttention(kernel_size=7)
        ],
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )

def get_multi_step_init_block(filters, name, top_n, router_temp, diversity_tau, steps, sr_ration):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
        "sr_ration": sr_ration
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouterMultiStep(
        branches=[
            ResidualBlock3x3(filters),
            ResidualBlock5x5(filters, groups=2),
            ResidualBlockDepthwise3x3(filters),
            ResidualBlockDepthwise5x5(filters),
            ResidualBlockDepthwise7x7(filters),
            ChannelSE(ratio=4),
            SpatialSE(),
            SpatialAttention(kernel_size=7),
            DummyBlock()
        ],
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n,
        steps=steps
    )


def get_multi_step_sparse_init_block(filters, name, top_n, router_temp, steps):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": 0.0,
        "explore_eps": 0,
        "ent_weight": 0.01,
        "load_balance_weight": 0.05,
        "route_temp": router_temp,
    }

    return SparseAdaptiveRouterMultiStep(
        branches=[
            ResidualBlock3x3(filters),
            ResidualBlock5x5(filters, groups=2),
            ResidualBlockDepthwise3x3(filters),
            ResidualBlockDepthwise5x5(filters),
            ResidualBlockDepthwise7x7(filters),
            ChannelSE(ratio=4),
            SpatialSE(),
            SpatialAttention(kernel_size=7),
            DummyBlock()
        ],
        router_reg=router_reg,
        router_settings=router_settings,
        dense_for_diversity=False,
        name=name,
        steps=steps
    )


def get_stem(branches, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=branches,
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )


def get_pooling(branches, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 64,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=branches,
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )


def get_block(branches, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=branches,
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )


def get_block(branches, name, top_n, router_temp, diversity_tau):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouter(
        branches=branches,
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n
    )

def get_multi_step_block(branches, name, top_n, router_temp, diversity_tau, steps):
    router_settings={
        "heads": 4,
        "dim_head": 32,
        "mlp_hidden": 64,
    }
    router_reg={
        "diversity_tau": diversity_tau,
        "explore_eps": 0,
        "ent_weight": 0,
        "load_balance_weight": 0,
        "route_temp": router_temp,
    }

    return AdaptiveRouterMultiStep(
        branches=branches,
        router_reg=router_reg,
        router_settings=router_settings,
        name=name,
        top_n=top_n,
        steps=steps
    )
