"""Team-shape possession model: p(home has the ball) from player shape only, ball hidden.

    model = load(cfg)                  # cfg["decoder_shape_model"]: the fitted JSON (results/tuning/team_shape/)
    p_home = predict(df, model)        # [N] float in (0, 1)

Features: for each name in FEATURES, the home-minus-away difference h_<name> - a_<name> of the
feature table (extract_features.py). adv_x_m, adv_vx and own_half_frac are attack-direction
normalised per team; the others (std_x_m, std_y_m, spread_m, mean_speed, walking_frac) need no
direction. No ball-derived column (ball_*, nearest_*, *_dmin_ball_m, *_d2_ball_m, *_n_within_*)
is used, so the model works where the ball is lost.

Model: antisymmetric logistic regression without intercept,
    p = sigmoid(sum_j coef_j * d_j / scale_j),   d = f_h - f_a,
scale only (no centring), so swapping the teams flips d and gives exactly 1 - p: no home bias can
be learned. A NaN feature contributes 0. Fitted on IDSSE tuning matches by
features/external/fit_team_shape.py (Bassek, Rein, Weber et al., Sci Data 12, 195 (2025),
doi:10.1038/s41597-025-04505-y, CC BY 4.0).
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ("adv_x_m", "adv_vx", "own_half_frac", "std_x_m", "std_y_m", "spread_m",
            "mean_speed", "walking_frac")
BALL_TOKENS = ("ball", "nearest", "within")    # any column naming one of these is ball-derived


def columns():
    """The feature-table columns the model reads."""
    return [p + f for f in FEATURES for p in ("h_", "a_")]


def design(df):
    """[N, len(FEATURES)] home-minus-away differences; NaN where either side is NaN."""
    return np.stack([df["h_" + f].to_numpy(float) - df["a_" + f].to_numpy(float)
                     for f in FEATURES], axis=1)


def scaled(d, scales_m):
    """d / scale, NaN -> 0 (no contribution)."""
    scales_m = np.asarray(scales_m, float)
    assert scales_m.shape == (d.shape[1],) and (scales_m > 0).all(), scales_m
    return np.nan_to_num(d / scales_m, nan=0.0)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))       # exact symmetry: sigmoid(-z) = 1 - sigmoid(z)


def predict(df, model):
    """p(home in possession) per row. model: dict with features, coef, scales."""
    assert tuple(model["features"]) == FEATURES, "model features {} != {}".format(
        model["features"], FEATURES)
    coef = np.asarray(model["coef"], float)
    return sigmoid(scaled(design(df), model["scales"]) @ coef)


def load(cfg):
    """The fitted model JSON named by cfg["decoder_shape_model"] (relative to the repo root)."""
    path = os.path.join(ROOT, cfg["decoder_shape_model"])
    with open(path) as f:
        model = json.load(f)
    assert model["status"] == "done", "{}: fit not finished".format(path)
    assert tuple(model["features"]) == FEATURES, "{}: features differ from team_shape.FEATURES".format(path)
    return model
