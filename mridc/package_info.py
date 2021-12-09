# Copyright (c) 2021, Dimitrios Karkalousos. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


MAJOR = 0
MINOR = 0
PATCH = 1
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "mridc"
__contact_names__ = "Dimitrios Karkalousos"
__contact_emails__ = "d.karkalousos@amsterdamumc.nl"
__homepage__ = "https://github.com/wdika/mridc"
__repository_url__ = "https://github.com/wdika/mridc"
__download_url__ = "https://github.com/wdika/mridc/releases"
__description__ = "Data Consistency for Magnetic Resonance Imaging"
__license__ = "Apache-2.0 License"
__keywords__ = (
    "machine-learning, deep-learning, compressed-sensing, pytorch, mri, medical-imaging, "
    "convolutional-neural-networks, unet, medical-image-processing, medical-image-analysis, "
    "data-consistency, mri-reconstruction, fastmri, recurrent-inference-machines, variational-network, "
    "cirim"
)
