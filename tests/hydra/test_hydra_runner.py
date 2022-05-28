# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/hydra/test_hydra_runner.py

import subprocess
import sys
from os import path

import pytest


class TestHydraRunner:
    @pytest.mark.integration
    def test_no_config(self):
        """ "Test app without config - fields missing causes error."""
        # Create system call.
        call = "python tests/hydra/my_app.py"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    @pytest.mark.integration
    def test_config1(self):
        """ "Test injection of valid config1."""
        # Create system call.
        call = "python tests/hydra/my_app.py --config-name config1.yaml"

        # Run the call as subprocess.
        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Make sure that .hydra dir is not present.
        assert not path.exists(".hydra")
        # Make sure that default hydra log file is not present.
        assert not path.exists("my_app.log")

    @pytest.mark.integration
    def test_config1_invalid(self):
        """ "Test injection of invalid config1."""
        # Create system call.
        call = "python tests/hydra/my_app.py --config-name config1_invalid.yaml"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    @pytest.mark.integration
    def test_config2(self):
        """ "Test injection of valid config2 from a different folder."""
        # Create system call.
        call = "python tests/hydra/my_app.py --config-name config2.yaml"

        # Run the call as subprocess.
        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Make sure that .hydra dir is not present.
        assert not path.exists(".hydra")
        # Make sure that default hydra log file is not present.
        assert not path.exists("my_app.log")

    @pytest.mark.integration
    def test_config2_invalid(self):
        """ "Test injection of invalid config2 from a different folder."""
        # Create system call.
        call = "python tests/hydra/my_app.py --config-path config_subdir --config-name config2_invalid.yaml"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    @pytest.mark.integration
    def test_config2_filepath_schema(self):
        """ "Test injection of valid config2 - using namepath with schema is prohibited."""
        # Create system call.
        call = "python tests/hydra/my_app.py --config-name config_subdir/config2_invalid.yaml"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)
