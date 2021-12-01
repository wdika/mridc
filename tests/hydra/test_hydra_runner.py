# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/blob/main/tests/hydra/test_hydra_runner.py

import subprocess
import sys
from os import path

import pytest


class TestHydraRunner:
    """Test the hydra runner."""

    @pytest.mark.integration
    def test_no_config(self):
        """Test app without config - fields missing causes error."""
        # Create system call.
        call = "python test/hydra/tmp_launch.py"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    @pytest.mark.integration
    def test_config1(self):
        """Test injection of valid config."""
        # Create system call.
        call = "python test/hydra/tmp_launch.py --config-name config.yaml"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Make sure that .hydra dir is not present.
        if path.exists(".hydra"):
            raise AssertionError
        # Make sure that default hydra log file is not present.
        if path.exists("tmp_launch.log"):
            raise AssertionError

    @pytest.mark.integration
    def test_config1_invalid(self):
        """Test injection of invalid config."""
        # Create system call.
        call = "python test/hydra/tmp_launch.py --config-name config_invalid.yaml"

        with pytest.raises(subprocess.CalledProcessError):
            # Run the call as subprocess.
            subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)
