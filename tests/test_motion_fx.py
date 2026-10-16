import numpy as np
import pytest

from xsuper import fx, motion
from xsuper.sketch import prepare_sketch
from xsuper.styles import STYLES


@pytest.mark.parametrize("camera", list(motion.CAMERA_PATHS))
def test_camera_never_shows_outside_the_drawing(camera):
    out_w, out_h = 1920, 1080
    gen_w, gen_h = motion.generation_size(camera)
    src = motion.prescale(np.zeros((gen_h, gen_w, 3), np.uint8), out_w, out_h)
    sh, sw = src.shape[:2]
    for m in motion.camera_matrices(camera, 48, sw, sh, out_w, out_h, focus=(sw * 0.8, sh * 0.2)):
        inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))
        for x, y in [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]:
            sx, sy, _ = inv @ [x, y, 1]
            assert -1 <= sx <= sw + 1 and -1 <= sy <= sh + 1, (camera, sx, sy)


def test_mouth_flaps_on_twos_and_close_in_silence():
    env = np.concatenate([np.ones(40), np.zeros(20)])
    track = motion.mouth_track(env, np.random.default_rng(0))
    assert track[:40].any() and not track[:40].all()   # flaps, not a frozen open mouth
    assert not track[40:].any()                         # shut when silent
    assert all(track[i] == track[i + 1] for i in range(0, 60, 2))  # changes only on twos


def test_on_twos_holds_frames():
    assert motion.on_twos(6, 10) == [0, 0, 2, 2, 4, 4]
    assert motion.on_twos(5, 2) == [0, 0, 1, 1, 1]


def test_grades_and_effects_keep_shape_and_type():
    img = np.random.default_rng(0).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    for style in STYLES.values():
        out = fx.grade_image(img, style.grade)
        assert out.shape == img.shape and out.dtype == np.uint8
    rng = np.random.default_rng(1)
    mask = fx.speed_lines_mask(160, 90, rng, (80, 45), 40)
    assert mask.shape == (90, 160) and mask.max() > 0
    blended = fx.overlay_mask(img, mask, (255, 255, 255))
    assert blended.shape == img.shape
    for v in range(3):
        assert set(np.unique(fx.impact_frame(img, v)).tolist()) <= {0, 18, 32, 205, 255}


def test_sketch_becomes_white_lines_on_black(demo):
    hint = np.asarray(prepare_sketch(demo / "sketches" / "leap.png", 1344, 768))
    assert hint.shape == (768, 1344, 3)
    assert hint.mean() < 60                 # mostly black background
    assert (hint > 200).mean() > 0.01       # with white strokes
