# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Parameter conversion."""


from io import BytesIO
from typing import Literal, cast

import numpy as np
import tenseal as ts

from .typing import NDArray, Parameters


def ndarrays_to_parameters(ndarrays: list[NDArray | ts.CKKSVector]) -> Parameters:
    """Convert NumPy ndarrays or TenSEAL vectors to parameters object.

    :param ndarrays: List of NumPy ndarrays or TenSEAL vectors

    :return: Pimped Flwr parameters object that contains list of serialized NumPy ndarrays / TenSEAL Vectors
        and a corresponding list that indicates this type.
    """
    parameters = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    tensors, tensor_type = zip(*parameters)
    return Parameters(tensors=list(tensors), tensor_type=list(tensor_type))


def parameters_to_ndarrays(
    parameters: Parameters, he_context: ts.Context | None = None
) -> list[NDArray | ts.CKKSVector]:
    """Convert parameters object to NumPy ndarrays or TenSEAL vectors.

    :param parameters: Pimped Flwr parameters object that contains list of serialized NumPy ndarrays / TenSEAL Vectors
        and a corresponding list that indicates this type.
    :param he_context: Homomorphic encryption context, to link the deserialized TenSEAL vectors to.

    :return: List of NumPy ndarrays or TenSEAL vectors.
    """
    return [
        bytes_to_ndarray(tensor, tensor_type, he_context)
        for tensor, tensor_type in zip(parameters.tensors, parameters.tensor_type)
    ]


def ndarray_to_bytes(
    ndarray: NDArray | ts.CKKSVector,
) -> tuple[bytes, Literal["numpy.ndarray", "tenseal.CKKSVector"]] | str:
    """Serialize NumPy ndarray or TenSEAL vector to bytes.

    :param ndarray: NumPy ndarray or TenSEAL vector

    :return: Serialized NumPy ndarray or TenSEAL vector and a corresponding string that indicates this type.
    """
    if isinstance(ndarray, ts.CKKSVector):
        return ndarray.serialize(), "tenseal.CKKSVector"
    else:
        bytes_io = BytesIO()
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(bytes_io, ndarray, allow_pickle=False)
        return bytes_io.getvalue(), "numpy.ndarray"


def bytes_to_ndarray(
    tensor: bytes, tensor_type: Literal["numpy.ndarray", "tenseal.CKKSVector"] | str, he_context: ts.Context | None = None
) -> NDArray | ts.CKKSVector:
    """Deserialize NumPy ndarray or TenSEAL vector from bytes.

    :param tensor: Serialized NumPy ndarray or TenSEAL vector.
    :param tensor_type: String that whether the tensor is a NumPy ndarray or TenSEAL vector.
    :param he_context: Homomorphic encryption context, to link the deserialized TenSEAL vectors to.

    :return: NumPy ndarray or TenSEAL vector
    """
    if tensor_type == "tenseal.CKKSVector":
        ndarray = ts.lazy_ckks_vector_from(tensor)
        ndarray.link_context(he_context)
        return ndarray
    else:
        bytes_io = BytesIO(tensor)
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
        return cast(NDArray, ndarray_deserialized)
