"""Definition of the action space.

File holds all classes which define the spline and with that also the action
definition of the problem.
"""
import logging
from abc import abstractmethod

from releso.exceptions import ParserException
from releso.util.logger import get_parser_logger
from releso.util.util_funcs import ModuleImportRaiser

try:
    from gustaf import NURBS, BSpline
except ImportError as err:
    BSpline = ModuleImportRaiser("gustaf", err)
    NURBS = ModuleImportRaiser("gustaf", err)
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.fields import PrivateAttr
from pydantic.types import confloat, conint

from releso.base_model import BaseModel


class VariableLocation(BaseModel):
    """Variable location class.

    Object of this class defines the position and movement possibilities for a
    single dimensions of a single control point of the spline.
    """
    #: coordinate of the current position of the current control point in the
    #: dimension
    current_position: float
    #: lower bound of possible values which can be reached by this variable
    min_value: Optional[float] = None
    #: upper bound  of possible values which can be reached by this variable
    max_value: Optional[float] = None
    #: number of steps the value range is divided into used to define the step
    #: if not given.
    n_steps: Optional[conint(ge=1)] = 10
    #: If discrete actions are used the step is used to define the new current
    #: position by adding/subtracting it from the current position.
    step: Optional[float] = None

    # non json variables
    #: Is true if min_value and max_value are not the same value
    _is_action: Optional[bool] = PrivateAttr(default=None)
    #: Original position needs to be saved so that the spline can be easily
    #: reset to its original state.
    _original_position: Optional[float] = PrivateAttr(default=None)

    def __init__(__pydantic_self__, **data: Any) -> None:
        """Constructor for VariableLocation."""
        super().__init__(**data)

    @validator("min_value", "max_value", always=True)
    @classmethod
    def set_variable_to_current_position_if_not_given(
            cls, v, values, field) -> float:
        """Validator for min_value and max_value.

        Validation of the min and max values for the current VariableLocation.
        If non are set no variability is assumed min = max = current_position

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Parser error if current_position is not already
            validated.

        Returns:
            float: value of the validated value.
        """
        if v is None:
            if "current_position" in values.keys():
                return values["current_position"]
            else:
                raise ParserException(
                    "VariableLocation", field,
                    "Please correctly define the current position.")
        return v

    @validator("max_value", always=True)
    @classmethod
    def max_value_is_greater_than_min_value(cls, v, values, field) -> float:
        """Validates that the max value is greater or equal to the min value.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Parser error if min_value is not already validated
            and if min value greater if max value.

        Returns:
            float: value of the validated value.
        """
        if "min_value" not in values.keys():
            raise ParserException(
                "VariableLocation", field, "Please define the min_value.")
        if v is None:
            raise RuntimeError("This should not have happened.")
        if v < values["min_value"]:
            raise ParserException(
                "VariableLocation", field,
                f"The min_value {values['min_value']} must be smaller or equal"
                f" to the max_value {v}.")
        return v

    @root_validator
    @classmethod
    def define_step(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validator for class.

        Validation function that defines the step taken if the action would be
        discrete. If nothing is given for the calculation a n_steps default
        value of 10 is used.

        Args:
            values (Dict[str, Any]): Validated variables of the object.

        Returns:
            Dict[str, Any]: Validated variables but if steps was not given
            before now it has a value.
        """
        step, n_steps, min_value, max_value = values.get("step"), values.get(
            "n_steps"), values.get("min_value"), values.get("max_value")
        value_range = max_value - min_value
        if step is not None:
            return values  # if step length is defined ignore n_steps
        elif n_steps is None:
            # if discrete but neither step nor n_steps is defined a default 10
            # steps are assumed.
            n_steps = 10
        step = value_range/n_steps
        values["step"] = step
        return values

    def is_action(self) -> bool:
        """Boolean if variable defines an action.

        Checks if the variable has variability. Meaning is there optimization
        between the min_value and the max_value.

        Returns:
            bool: True if (self.max_value > self.current_position) or
            (self.min_value < self.current_position)
        """
        if self._is_action is None:
            self._is_action = (self.max_value > self.current_position) or (
                self.min_value < self.current_position)
        return self._is_action

    def apply_discrete_action(self, increasing: bool) -> float:
        """Apply a discrete action to the variable.

        Apply the a discrete action to the current value. Also applies clipping
        if min/max value is surpassed.

        Args:
            increasing (bool): Toggle whether or not the value should increase
            or not.

        Returns:
            float: Value of the position that the current position is now at.
        """
        if self._original_position is None:
            self._original_position = self.current_position
        step = self.step if increasing else -self.step
        self.current_position = np.clip(
            self.current_position + step, self.min_value, self.max_value)
        return self.current_position

    def apply_continuos_action(self, value: float) -> float:
        """Apply a continuous action the variable.

        Apply the zero-mean normalized value as the new current position.
        Needs to descale the value first. Also applies clipping.

        Args:
            value (float): Scaled value [-1,1] that needs to be descaled and
            applied as the new current position.

        Returns:
            float: New clipped value.
        """
        if self._original_position is None:
            self._original_position = self.current_position
        delta = self.max_value - self.min_value
        descaled_value = ((value+1.)/2.) * delta
        self.current_position = np.clip(
            descaled_value+self.min_value, self.min_value, self.max_value)
        return self.current_position

    def reset(self) -> None:
        """Resets the current_location to the initial position."""
        if self._original_position is not None:
            self.current_position = self._original_position


class SplineSpaceDimension(BaseModel):
    """Defines a single spline space dimension of the current spline.

    The dimension is a dimension of the parametric spline dimension.
    """
    #: Name of the spline dimension used for easier identification
    name: str
    #: number of control points in this dimension
    number_of_points: conint(ge=1)
    #: degree of the spline in this dimension
    degree: conint(ge=1)
    #: knot vector describing the knot intervals of the spline in the current
    #: dimension
    knot_vector: Optional[List[float]]

    @validator("knot_vector", always=True)
    @classmethod
    def validate_knot_vector(
            cls, v: Optional[List[float]],
            values: Dict[str, Any]) -> List[float]:
        """Validator for knot_vector.

        If knot vector not given tries to make a default open knot vector.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Emitted parser error.

        Returns:
            float: value of the validated value.
        """
        if "number_of_points" in values.keys() and "degree" in values.keys() \
                and "name" in values.keys():
            n_knots = values["number_of_points"] + values["degree"] + 1
        else:
            raise ParserException(
                "SplineSpaceDimension", "knot_vector", "During validation the "
                "prerequisite variables number_of_points and degree were not "
                "present.")
        if type(v) is list:
            get_parser_logger().debug(
                f"The knot_vector for dimension {values['name']} is given.")
            if len(v) == n_knots:
                return v
            else:
                get_parser_logger().warning(
                    f"The knot vector does not contain the correct number of "
                    f"items for an open knot vector definition. (is: {len(v)},"
                    f" should_be: {n_knots}).")
        elif v is None:
            get_parser_logger().debug(
                f"The knot_vector for dimension {values['name']} is not given,"
                f" trying to generate one in the open format.")
            starting_ending = values["degree"] + 1
            middle = n_knots - (2 * starting_ending)
            if middle >= 0:
                knot_vec = list(np.append(np.append(np.zeros(
                    starting_ending-1), np.linspace(0, 1, middle+2)),
                    np.ones(starting_ending-1)))
            else:
                get_parser_logger().warning(
                    f"The knot vector is shorter {n_knots} than the length "
                    f"given by the open format {starting_ending*2}. Knot "
                    "vector is created by adding the starting and ending "
                    "parts. The knot vector might be to long.")
                knot_vec = list(
                    np.array(
                        [
                            np.zeros(starting_ending),
                            np.ones(starting_ending)]).flatten())
            return knot_vec

    def get_knot_vector(self) -> List[float]:
        """Function returns the Knot vector given in the object.

        Returns:
            List[float]: Direct return of the inner variable.
        """
        return self.knot_vector


class SplineDefinition(BaseModel):
    """Defines the spline.

    Base class for the NURBS and B-Spline implementations.
    """
    #: Definition of the space dimensions of the spline
    space_dimensions: List[SplineSpaceDimension]
    #: Non parametric spline dimensions is currently not used.
    spline_dimension: conint(ge=1)
    #: control point grid of the spline needs to be converted into an
    #: numpy.ndarray for 3D examples.
    control_point_variables: Optional[List[List[Union[
        VariableLocation, confloat(ge=0, le=1)]]]]

    @validator("control_point_variables", always=True)
    @classmethod
    def make_default_control_point_grid(
            cls, v, values) -> List[List[VariableLocation]]:
        """Validator for control_point_variables.

        If value is None a equidistant grid of control points will be given
        with variability of half the space between the each control point.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Emitted parser error.

        Returns:
            List[List[VariableLocation]]: Definition of the control_points
        """
        if v is None:
            if "space_dimensions" not in values.keys():
                raise ParserException(
                    "SplineDefinition", "control_point_variables",
                    "During validation the prerequisite variable "
                    f"space_dimensions was not present."+str(values))
            spline_dimensions = values["space_dimensions"]
            n_points_in_dim = [
                dim.number_of_points for dim in spline_dimensions]
            n_points_in_dim.reverse()
            # this can be done in a smaller and faster footprint but for
            # readability and understanding
            dimension_lists = []
            dim_spacings = []
            # create value range in each dimension separately
            for n_p in n_points_in_dim:
                dimension_lists.append(list(np.linspace(0., 1., n_p)))
                dim_spacings.append(1./((n_p-1)*2))
            # create each control point for by concatenating each value in each
            # list with each value of all other lists
            save_location = str(values["save_location"])
            for inner_dim, dim_spacing in zip(dimension_lists, dim_spacings):
                # first iteration the v vector must be initialized with the
                # first dimension vector
                if v is None:
                    v = []
                    for element in inner_dim:
                        if element == 0.:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element,
                                    max_value=element+dim_spacing,
                                    save_location=save_location))
                        elif element == 1.:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element - dim_spacing,
                                    max_value=element,
                                    save_location=save_location))
                        else:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element - dim_spacing,
                                    max_value=element+dim_spacing,
                                    save_location=save_location))
                # for each successive dimension for each existing element in v
                # each value in the new dimension must be added
                else:
                    temp_v = []
                    for current_list in v:
                        for element in inner_dim:
                            elem = copy.deepcopy(current_list if type(
                                current_list) is list else [current_list])
                            if element == 0.:
                                temp_v.append(
                                    [VariableLocation(
                                        current_position=element,
                                        min_value=element,
                                        max_value=element+dim_spacing,
                                        save_location=save_location)] + elem)
                            elif element == 1.:
                                temp_v.append(
                                    [VariableLocation(
                                        current_position=element,
                                        min_value=element - dim_spacing,
                                        max_value=element,
                                        save_location=save_location)] + elem)
                            else:
                                temp_v.append(
                                    [VariableLocation(
                                        current_position=element,
                                        min_value=element - dim_spacing,
                                        max_value=element+dim_spacing,
                                        save_location=save_location)] + elem)
                    v = temp_v
        return v

    @validator("control_point_variables", each_item=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(
            cls, v, values):
        """Validator for control_point_variables.

        Converts all control points values into VariableLocations if the value
        is given as a simple float. Simple float will be converted into
        VariableLocation with current_position=value and no variability.

        Args:
            v ([type]): value to validate

        Returns:
            [type]: validated value
        """
        new_list = []
        for element in v:
            new_list.append(
                VariableLocation(
                    current_position=element,
                    save_location=values["save_location"]
                ) if type(element) is float else element)
            if not 0. <= new_list[-1].current_position <= 1.:
                raise ParserException(
                    "SplineDefinition", "control_point_variables",
                    "The control_point_variables need to be inside an unit "
                    "hypercube. Found a values outside this unit hypercube.")
        return new_list

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Spline.

        Currently the number of points in the spline is calculated by
        multiplying the number of points in each spline dimension.

        Returns:
            int: number of points in the spline
        """
        return np.prod(
            [dimension.number_of_points for dimension in self.space_dimensions]
        )

    def get_control_points(self) -> List[List[float]]:
        """Returns the positions of all control points in a two deep list.

        Returns:
            List[List[float]]: Positions of all control points.
        """
        return [
            [control_point.current_position for control_point in sub_list]
            for sub_list in self.control_point_variables]

    def get_actions(self) -> List[VariableLocation]:
        """Return all actions this spline defines.

        Returns list of VariableLocations but only if the variable location is
        actually variable.

        Returns:
            List[VariableLocation]: See above
        """
        self.get_logger().debug("Collecting all actions for BSpline.")
        return [
            variable for sub_dim in self.control_point_variables
            for variable in sub_dim if variable.is_action()]

    @abstractmethod
    def get_spline(self) -> Any:
        """Generates the current Spline in the gustaf format.

        Notes:
            This is an abstract method.

        Returns:
            Spline: Spline that is generated.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the position of all DOF this Spline defines.

        Resets all control points in the spline to its original position
        (position they were initialized with).
        """
        for control_point in self.control_point_variables:
            for dim in control_point:
                dim.reset()


class BSplineDefinition(SplineDefinition):
    """Definition of the BSpline implementation of the ReLeSO Toolbox."""

    def get_spline(self) -> BSpline:
        """Creates the BSpline from the definition given by the json file.

        Returns:
            BSpline: given by the #degrees and knot_vector in each
            space_dimension and the current control points.
        """
        self.get_logger().debug("Creating Gustaf BSpline.")
        self.get_logger().debug(
            f"With control_points: {self.get_control_points()}")

        return BSpline(
            [space_dim.degree for space_dim in self.space_dimensions],
            [space_dim.get_knot_vector()
             for space_dim in self.space_dimensions],
            self.get_control_points()
        )

    def draw_action_space(
            self, save_location: Optional[str] = None, no_axis: bool = False,
            fig_size: List[float] = [6, 6], dpi: int = 400):
        """Draw the spline control points and the play they have.

        Currently only available for B-Splines with a 2 parametric dimensions.

        Args:
            save_location (Optional[str], optional): Save location of the
            produced file. Defaults to None.
            no_axis (bool, optional): Whether or not the axis of the plot are
            displayed. Defaults to False.
            fig_size (List[float], optional): Size of the produced figure in
            inches. Defaults to [6, 6].
            dpi (int, optional): DPI of the produced image. Defaults to 400.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        control_points = self.control_point_variables
        if not len(self.space_dimensions) == 2:
            raise RuntimeError(
                "Could not draw the splines action space. Only a 2D parametric"
                " space is currently available.")
        phi = np.linspace(0, 2*np.pi, len(control_points))
        rgb_cycle = np.vstack((            # Three sinusoids
            .5*(1.+np.cos(phi)),  # scaled to [0,1]
            .5*(1.+np.cos(phi+2*np.pi/3)),  # 120° phase shifted.
            .5*(1.+np.cos(phi-2*np.pi/3)))).T  # Shape = (60,3)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        dots = [[], []]

        for elem, color in zip(control_points, rgb_cycle):
            dots[0].append(elem[0].current_position)
            dots[1].append(elem[1].current_position)
            cur_pos = np.array([dots[0][-1], dots[1][-1]])
            spanning_elements = []
            no_boundary = False
            for item in elem:
                spanning_elements.append([item.min_value, item.max_value])
                if item.min_value == item.max_value:
                    no_boundary = True
            boundary = []
            for i, j in zip([0, 1, 1, 0], [0, 0, 1, 1]):
                end_pos = np.array(
                    [spanning_elements[0][i], spanning_elements[1][j]])
                if not np.isclose(cur_pos, end_pos).all():  # draw arrow
                    difference = end_pos-cur_pos
                    ax.arrow(cur_pos[0], cur_pos[1], difference[0] *
                             0.9, difference[1]*0.9, width=0.005, color=color)
                boundary.append(end_pos)
            if not no_boundary:
                pol = Polygon(boundary, facecolor=color,
                              linewidth=1, alpha=0.2)
                ax.add_patch(pol)
        ax.scatter(dots[0], dots[1], c=rgb_cycle, marker="o", s=50, zorder=3)
        if no_axis:
            plt.axis("off")
        if save_location:
            fig.savefig(save_location, transparent=True)
            plt.close()


class NURBSDefinition(SplineDefinition):
    """Definition of a NURBS spline.

    Definition of the NURBS implementation of the ReLeSO Toolbox, in
    comparison to the B-Spline implementation only an additional weights
    vector is added.
    """
    #: weights for the NURBS Spline definition. Other parameters are part of
    #: the spline definition class. Can be fixed or changeable
    weights: List[Union[float, VariableLocation]]

    @validator("weights")
    @classmethod
    def validate_weights(
            cls, v: Optional[List[Union[float, VariableLocation]]],
            values: Dict[str, Any]) -> List[Union[float, VariableLocation]]:
        """Validator for variable weights.

        Validate if the correct number of weights are present in the weight
        vector. If weight vector is not given weight vector should be all ones.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate
            values (Dict[str, Any]): Previously validated variables

        Raises:
            ParserException: Parser Error

        Returns:
            List[Union[float, VariableLocation]]: Filled weight vector.
        """
        if "space_dimensions" not in values.keys():
            raise ParserException(
                "SplineDefinition", "weights",
                "During validation the prerequisite variable space_dimensions "
                "were not present.")
        n_cp = np.prod(
            [
                space_dim.number_of_points for space_dim in
                values["space_dimensions"]])
        if type(v) is list:
            if len(v) == n_cp:
                get_parser_logger().debug(
                    "Found correct number of weights in SplineDefinition.")
            else:
                raise ParserException(
                    "SplineDefinition NURBS", "weights", f"The length of the "
                    f"weight vector {len(v)} is not the same as the number of "
                    f"control_points {n_cp}.")
        elif v is None:
            v = list(np.ones(n_cp))
        return v

    @validator("weights", each_item=True)
    @classmethod
    def convert_weights_into_variable_location(
            cls, v: List[Union[float, VariableLocation]]
    ) -> List[VariableLocation]:
        """Validator for variable weights.

        Convert all float values in the weight vector into VariableLocations.
        So that these can also be used as actions.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate

        Returns:
            List[VariableLocation]: Filled weight vector.
        """
        if type(v) is float:
            return VariableLocation(current_position=v)
        return v

    def get_spline(self) -> NURBS:
        """Creates a NURBSSpline from the definition given by the json file.

        Returns:
            NURBS: NURBSSpline
        """
        self.get_logger().debug("Creating Gustaf NURBS.")

        return NURBS(
            [space_dim.degree for space_dim in self.space_dimensions],
            [space_dim.get_knot_vector()
             for space_dim in self.space_dimensions],
            self.get_control_points(),
            self.weights
        )

    def get_actions(self) -> List[VariableLocation]:
        """Extends the control point actions with the weight actions.

        Returns:
            List[VariableLocation]: List of possible actions for this NURBS
            spline.
        """
        actions = super().get_actions()
        actions.extend(
            [variable for variable in self.weights if variable.is_action()])
        return actions

    def reset(self) -> None:
        """Resets the spline to the original shape."""
        super().reset()
        for weight in self.weights:
            weight.reset()


class CubeDefinition(BaseModel):
    """Defines a simple cube. TODO make better docu!"""
    control_points: List[List[VariableLocation]]

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Cube.

        Number of control points multiplied bz the number of dimensions for
        each cp.

        Returns:
            int: number of points in the spline
        """
        return int(len(self.control_points)*len(self.control_points[0]))

    @validator("control_points", each_item=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(
            cls, v, values):
        """Validator control_points.

        Converts all control points values into VariableLocations if the value
        is given as a simple float. Simple float will be converted into
        VariableLocation with current_position=value and no variability.

        Args:
            v ([type]): value to validate

        Returns:
            [type]: validated value
        """
        new_list = []
        for element in v:
            new_list.append(
                VariableLocation(
                    current_position=element,
                    save_location=values["save_location"]
                ) if type(element) is float else element)
            if not 0. <= new_list[-1].current_position <= 1.:
                raise ParserException(
                    "SplineDefinition", "control_point_variables",
                    "The control_point_variables need to be inside an unit "
                    "hypercube. Found a values outside this unit hypercube.")
        return new_list

    def get_control_points(self) -> List[List[float]]:
        """Returns the positions of all control points in a two deep list.

        Returns:
            List[List[float]]: Positions of all control points.
        """
        return [
            [control_point.current_position for control_point in sub_list]
            for sub_list in self.control_points]

    def get_actions(self) -> List[VariableLocation]:
        """Returns the action defined through this Cube.

        Returns list of VariableLocations but only if the variable location is
        actually variable.

        Returns:
            List[VariableLocation]: See above
        """
        self.get_logger().debug("Collecting all actions for Cube.")
        return [
            variable for sub_dim in self.control_points
            for variable in sub_dim if variable.is_action()]

    def get_spline(self) -> Any:
        """Generates the current Spline in the gustaf format.

        Notes:
            This is an abstract method.

        Returns:
            Spline: Spline that is generated.
        """
        return self.get_control_points()

    def reset(self) -> None:
        """Resets the spline to the original shape."""
        for cp in self.control_points:
            for variable in cp:
                variable.reset()

    def draw_action_space(
            self, save_location: Optional[str] = None, no_axis: bool = False,
            fig_size: List[float] = [6, 6], dpi: int = 400):
        """Draw the action space of the defined Cube.

        Args:
            save_location (Optional[str], optional): _description_.
            Defaults to None.
            no_axis (bool, optional): _description_. Defaults to False.
            fig_size (List[float], optional): _description_. Defaults to [6, 6].
            dpi (int, optional): _description_. Defaults to 400.

        Raises:
            RuntimeError: _description_
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        control_points = self.control_points
        if len(self.control_points[0]) > 2:
            raise RuntimeError(
                "Could not draw the splines action space. Only a 2D parametric"
                " space is currently available.")
        elif len(self.control_points[0]) == 1:
            for idx, cp in enumerate(control_points):
                cp.insert(
                    0,
                    VariableLocation(
                        current_location=float(idx),
                        save_location=self.save_location
                    )
                )
        phi = np.linspace(0, 2*np.pi, len(control_points))
        rgb_cycle = np.vstack((            # Three sinusoids
            .5*(1.+np.cos(phi)),  # scaled to [0,1]
            .5*(1.+np.cos(phi+2*np.pi/3)),  # 120° phase shifted.
            .5*(1.+np.cos(phi-2*np.pi/3)))).T  # Shape = (60,3)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        dots = [[], []]

        for elem, color in zip(control_points, rgb_cycle):
            dots[0].append(elem[0].current_position)
            dots[1].append(elem[1].current_position)
            cur_pos = np.array([dots[0][-1], dots[1][-1]])
            spanning_elements = []
            no_boundary = False
            for item in elem:
                spanning_elements.append([item.min_value, item.max_value])
                if item.min_value == item.max_value:
                    no_boundary = True
            boundary = []
            for i, j in zip([0, 1, 1, 0], [0, 0, 1, 1]):
                end_pos = np.array(
                    [spanning_elements[0][i], spanning_elements[1][j]])
                if not np.isclose(cur_pos, end_pos).all():  # draw arrow
                    difference = end_pos-cur_pos
                    ax.arrow(cur_pos[0], cur_pos[1], difference[0] *
                             0.9, difference[1]*0.9, width=0.005, color=color)
                boundary.append(end_pos)
            if not no_boundary:
                pol = Polygon(boundary, facecolor=color,
                              linewidth=1, alpha=0.2)
                ax.add_patch(pol)
        ax.scatter(dots[0], dots[1], c=rgb_cycle, marker="o", s=50, zorder=3)
        if no_axis:
            plt.axis("off")
        if save_location:
            fig.savefig(save_location, transparent=True)
            plt.close()


# should always be a derivate of SplineDefinition
SplineTypes = Union[NURBSDefinition, BSplineDefinition, CubeDefinition]


class Spline(BaseModel):
    """Definition of the spline.

    Can be deleted in the next round of reworks.

    Had a historical significance but had all its additional functions removed.
    Please remove in next rework.
    """
    spline_definition: SplineTypes

    def get_control_points(self) -> List[List[float]]:
        """Return all control_points of the spline.

        Returns:
            List[List[float]]: Nested list of control_points.
        """
        return self.spline_definition.get_control_points()

    def get_spline(self) -> SplineTypes:
        """Creates the current spline.

        Returns:
            Union[SplineTypes]: [description]
        """
        return self.spline_definition.get_spline()

    def get_actions(self) -> List[VariableLocation]:
        """Return actions defined through the used spline.

        Returns a list of VariableLocations that actually have a defined
        variability and can therefor be an action.

        Returns:
            List[VariableLocation]: VariableLocations that have 'wriggle' room.
        """
        return self.spline_definition.get_actions()

    def reset(self) -> None:
        """Resets the spline to its initial values."""
        self.spline_definition.reset()
