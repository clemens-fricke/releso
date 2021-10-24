from copy import deepcopy
import logging
from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment
from SbSOvRL.validation import Validation
from SbSOvRL.callback import EpisodeLogCallback
from SbSOvRL.verbosity import Verbosity
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.exceptions import SbSOvRLValidationNotSet
from SbSOvRL.parser_environment import Environment
from pydantic import validator
from typing import Union, Optional
from pydantic.fields import Field, PrivateAttr
from pydantic.types import conint
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.base_class import BaseAlgorithm
import pathlib
import datetime
import numpy as np

class BaseParser(SbSOvRL_BaseModel):
    verbosity: Verbosity = Field(default_factory=Verbosity)
    agent: AgentTypeDefinition
    environment: Union[Environment]
    number_of_timesteps: conint(ge=1)
    number_of_episodes: conint(ge=1)
    save_location: Optional[str]
    validation: Optional[Validation]


    # internal objects
    _agent: Optional[BaseAlgorithm] = PrivateAttr(default=None)

    @validator("save_location")
    @classmethod
    def add_datetime_to_save_location(cls, v) -> str:
        if '{}' in v:
            v = v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        path = pathlib.Path(v).expanduser().resolve()
        return path
        
    def learn(self) -> None:
        """
        Starts the training that is specified in the loaded json file.
        """
        self._agent = self.agent.get_agent(self.environment.get_gym_environment())

        callbacks = [
            StopTrainingOnMaxEpisodes(max_episodes=self.number_of_episodes),
            EpisodeLogCallback(episode_log_location=self.save_location/"episode_log.csv", verbose=1)
        ]
        validation_environment = self._create_validation_environment()

        if self.validation:
            if self.validation.should_add_callback():
                callbacks.append(self.validation.get_callback(validation_environment.get_gym_environment(), save_location=self.save_location))
        

        print(self.number_of_timesteps)
        self._agent.env.reset()
        self._agent.learn(self.number_of_timesteps, callback=callbacks, tb_log_name=self.agent.get_next_tensorboard_experiment_name(), reset_num_timesteps=False)
        self.save_model()
        self.evaluate_model(validation_environment)

    def export_spline(self, file_name: str) -> None:
        """Exports the current spline to the specified location.

        Args:
            file_name (str): Path to the location where the spline should be stored at.
        """
        self.environment.export_spline(file_name)

    def export_mesh(self, file_name:str) -> None:
        """Exports the current mesh to the specified location.

        Args:
            file_name (str): Path to the location where the mesh should be stored at.
        """
        self.environment.export_mesh(file_name)

    def save_model(self, file_name: Optional[str] = None) -> None:
        """Saves the current agent to the specified location or to a default location.

        Args:
            file_name (Optional[str]): Path where the agent is saved to. If None will see if json had a save location if also not given will use a default location. Defaults to None.
        """
        if file_name is None and self.save_location is not None:
            path = pathlib.Path(self.save_location)
        elif file_name is not None:
            path = pathlib.Path(file_name)
        else:
            path = pathlib.Path(f"default_model_safe_location/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(path/"model_end.save")
            
    def evaluate_model(self, validation_env: Union[None, Environment] = None, throw_error_if_None: bool = False) -> None:
        """Evaluate the model with the parameters defined in the validation variable. If an agent is already loaded use this agent, else get the agent from the agent variable. Validation will be done inside the 

        Args:
            validation_env (Union[None, Environment], optional): If validation environment already exists it will be used else a new validation evironment will be created. Defaults to None.
            throw_error_if_None (bool, optional): If this is set and the validation variable is None an error is thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet: Thrown if validation is absolutly needed. If not absolutly needed the validation will not be done but no error will be thrown.
        """
        if self.validation:
            if validation_env is None:
                validation_env = self._create_validation_environment(True)
            if self._agent is None:
                self._agent = self.agent.get_agent(validation_env.get_gym_environment())
            reward, episode_length = self.validation.end_validation(agent=self._agent, environment=validation_env.get_gym_environment())
            reward_array = np.array(reward)
            mean_reward = reward_array.mean()
            reward_std = reward_array.std()
            log_str = f"The end validation had the following results: mean_reward = {mean_reward}; reward_std = {reward_std}"
            logging.getLogger("SbSOvRL_environment").info(log_str)
            print("The reward per episode was:", reward)
            print("The length per episode was:", episode_length)
        elif throw_error_if_None:
            raise SbSOvRLValidationNotSet()

    def _create_validation_environment(self, throw_error_if_None: bool = False) -> Environment:
        """Creates a validation environment.

        Args:
            throw_error_if_None (bool, optional): If this is set and the validation variable is None an error is thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet: Thrown if validation is absolutly needed. If not absolutly needed the validation will not be done but no error will be thrown.

        Returns:
            Environment: Validation environment. Is not a 'gym' environment but and SbSOvRL.parser_environment.Environment. Create the gym environment by calling the function env.get_gym_environment()
        """
        if self.validation is None:
            if throw_error_if_None:
                raise SbSOvRLValidationNotSet()
            return None
        validation_environment = deepcopy(self.environment)
        validation_environment.set_validation(**self.validation.get_environment_validation_parameters(self.save_location))
        return validation_environment
