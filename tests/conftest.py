import builtins
import os
import pathlib

import gymnasium as gym
import pytest
import requests


class Dummy_Environment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.episode_length = 10
        self.episode_counter = 0

    def step(self, action):
        self.episode_counter += 1
        return (
            [sum(action)],
            self.episode_counter,
            self.episode_counter >= self.episode_length,
            False,
            {},
        )

    def reset(self, **kwargs):
        self.episode_counter = 0
        return [0], {}


@pytest.fixture
def provide_dummy_environment():
    return Dummy_Environment()


@pytest.fixture
def clean_up_provider():
    def recursive_file_remove(path):
        """Remove a file or directory and its contents.

        Author: GitHubCopilot (14.08.2023)
        """
        if not path.exists():
            return
        if path.is_file():
            path.unlink()
            return
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                recursive_file_remove(child)
        path.rmdir()

    return recursive_file_remove


@pytest.fixture
def hide_available_import(monkeypatch):
    """Hide the available import from the user.

    This is used to test the import of the available module.

    Author:
        https://stackoverflow.com/a/60229056
    """
    import_orig = builtins.__import__

    def mock_import_available(name, *args, **kwargs):
        # raise RuntimeError(name)
        with open("import.txt", "a") as file:
            file.write(f"{name}\n")
        if name == "splinepy.helpme.ffd":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_available)


def dir_save_location_path(make_dir=False):
    dir_save_location = (
        pathlib.Path(__file__).parent / "test_save_location_please_delete"
    ).resolve()
    if make_dir:
        dir_save_location.mkdir(parents=True, exist_ok=True)
    return dir_save_location


@pytest.fixture
def dir_save_location():
    path = dir_save_location_path()
    yield path
    if os.path.isdir(path):
        os.rmdir(path)


@pytest.fixture
def load_sample_file(request):
    file_name = request.param
    base_url = "https://raw.githubusercontent.com/tataratat/samples/main/"
    local_path = pathlib.Path(__file__).parent / "samples/"
    local_file = local_path / file_name
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if not local_file.is_file():
        url = base_url + file_name
        # print(f"Downloading {url} to {local_file}")
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Could not download {url}")
        with open(local_file, "wb") as file:
            file.write(response.content)
    return local_file


@pytest.fixture
def dummy_file():
    file_name = pathlib.Path("dummy_file_please_delete.txt")
    file_name.touch()
    yield file_name
    if file_name.is_file():
        file_name.unlink()
