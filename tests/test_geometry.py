import numpy as np
import pytest
from conftest import dir_save_location_path
from gymnasium import spaces
from pydantic import ValidationError

from releso.geometry import FFDGeometry, Geometry
from releso.shape_parameterization import ShapeDefinition
from releso.spline import BSplineDefinition, NURBSDefinition
from releso.util.logger import VerbosityLevel


@pytest.mark.parametrize(
    (
        "shape_definition",
        "action_based",
        "discrete_action",
        "reset_random",
        "correct_type",
        "n_action_variables",
    ),
    [
        (
            "default_shape",
            None,
            None,
            None,
            ShapeDefinition,
            2,
        ),  # test that default is correct
        ("default_shape", True, True, True, ShapeDefinition, 2),
        ("default_shape", False, False, False, ShapeDefinition, 2),
        ("bspline_shape", None, False, None, BSplineDefinition, 18),
        ("nurbs_shape", None, False, None, NURBSDefinition, 20),
    ],
)
def test_geometry_init(
    shape_definition,
    action_based,
    discrete_action,
    reset_random,
    correct_type,
    n_action_variables,
    dir_save_location,
    caplog,
    request,
):
    call_dict = {
        "shape_definition": request.getfixturevalue(shape_definition),
        "save_location": dir_save_location,
    }
    if action_based is not None:
        call_dict["action_based_observation"] = action_based
    if discrete_action is not None:
        call_dict["discrete_actions"] = discrete_action
    if reset_random is not None:
        call_dict["reset_with_random_action_values"] = reset_random
    geometry = Geometry(**call_dict)
    assert isinstance(geometry.shape_definition, correct_type)
    assert geometry.action_based_observation == (
        action_based if action_based is not None else True
    )
    assert geometry.discrete_actions == (
        discrete_action if discrete_action is not None else True
    )
    assert geometry.reset_with_random_action_values == (
        reset_random if reset_random is not None else False
    )

    # check get all actions
    geometry.setup("id")

    # control points
    assert (
        geometry.get_control_points()
        == geometry.shape_definition.get_control_points()
    )

    assert geometry.is_geometry_changed() == False
    assert geometry.apply() == geometry.get_control_points()

    original_cps = geometry.get_control_points()

    # actions
    assert len(geometry._actions) == len(
        geometry.shape_definition.get_actions()
    )
    assert len(geometry._actions) == n_action_variables

    act_def = geometry.get_action_definition()
    if geometry.discrete_actions:
        assert isinstance(act_def, spaces.Discrete)
        assert act_def.n == 2 * len(geometry._actions)
        geometry.apply_action(3)
    else:
        assert isinstance(act_def, spaces.Box)
        assert act_def.shape == (len(geometry._actions),)
        geometry.apply_action(np.random.rand(len(geometry._actions)))
    assert geometry.is_geometry_changed() == True

    # observation
    if geometry.action_based_observation:
        assert geometry.get_observation_definition()[1].shape == (
            len(geometry._actions),
        )
        assert len(geometry.get_observation()) == len(geometry._actions)
    else:
        with caplog.at_level(VerbosityLevel.WARNING):
            geometry.get_observation_definition()
            assert (
                "Observation space is accessed which should not happen."
                in caplog.text
            )
            assert geometry.get_observation() is None

    # reset
    current_cps = geometry.get_control_points()
    assert current_cps != original_cps
    if geometry.reset_with_random_action_values:
        reset_cps = geometry.reset()
        assert reset_cps != current_cps
        assert reset_cps != original_cps
    else:
        assert geometry.reset() == original_cps

    # random action with seed produces same result
    set_1 = geometry.apply_random_action("asd")
    set_2 = geometry.apply_random_action("asd")
    assert np.allclose(set_1, set_2)


@pytest.mark.parametrize(
    (
        "shape_definition",
        "action_based",
        "discrete_action",
        "reset_random",
        "load_sample_file",
        "export_mesh",
        "error",
    ),
    [
        (
            "default_shape",
            False,
            False,
            False,
            "volumes/tet/3DBrickTet.msh",
            None,
            "FFD can only be performed with a Gustaf Spline",
        ),
        (
            "bspline_shape",
            None,
            False,
            None,
            "faces/quad/2DChannelQuad.msh",
            {"format": "mixd", "export_path": "test"},
            False,
        ),
        (
            "nurbs_shape",
            None,
            False,
            None,
            "faces/quad/2DChannelQuad.msh",
            None,
            False,
        ),
    ],
    indirect=["load_sample_file"],
)
def test_ffd_geometry_init(
    shape_definition,
    action_based,
    discrete_action,
    reset_random,
    load_sample_file,
    export_mesh,
    error,
    dir_save_location,
    caplog,
    request,
):
    call_dict = {
        "shape_definition": request.getfixturevalue(shape_definition),
        "save_location": dir_save_location,
        "mesh": {
            "path": load_sample_file,
            "save_location": dir_save_location,
            "dimensions": 2,
        },
    }
    if export_mesh is not None:
        export_mesh["save_location"] = dir_save_location
        call_dict["export_mesh"] = export_mesh
    if action_based is not None:
        call_dict["action_based_observation"] = action_based
    if discrete_action is not None:
        call_dict["discrete_actions"] = discrete_action
    if reset_random is not None:
        call_dict["reset_with_random_action_values"] = reset_random

    if error:
        with pytest.raises(ValidationError) as err:
            geometry = Geometry(**call_dict)
            assert error in str(err.value)
        return

    geometry = FFDGeometry(**call_dict)
    assert geometry.action_based_observation == (
        action_based if action_based is not None else True
    )
    assert geometry.discrete_actions == (
        discrete_action if discrete_action is not None else True
    )
    assert geometry.reset_with_random_action_values == (
        reset_random if reset_random is not None else False
    )

    with caplog.at_level(VerbosityLevel.INFO):
        # This warning should happen
        geometry.setup("id")
        assert "Found empty dimension" in caplog.text
    # control points
    assert (
        geometry.get_control_points()
        == geometry.shape_definition.get_control_points()
    )

    assert geometry.is_geometry_changed() == False
    geometry.apply()

    # assert geometry.apply() == geometry.get_control_points()

    original_cps = geometry.get_control_points()

    # actions
    assert len(geometry._actions) == len(
        geometry.shape_definition.get_actions()
    )
    act_def = geometry.get_action_definition()
    if geometry.discrete_actions:
        assert isinstance(act_def, spaces.Discrete)
        assert act_def.n == 2 * len(geometry._actions)
        geometry.apply_action(3)
    else:
        assert isinstance(act_def, spaces.Box)
        assert act_def.shape == (len(geometry._actions),)
        geometry.apply_action(np.random.rand(len(geometry._actions)))
    assert geometry.is_geometry_changed() == True
