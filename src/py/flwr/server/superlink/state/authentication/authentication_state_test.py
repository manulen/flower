# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Test for authentication state."""


from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    compute_hmac,
    generate_key_pairs,
    generate_shared_key,
    public_key_to_bytes,
    verify_hmac,
)

from .in_memory_auth_state import InMemoryAuthState
from .sqlite_auth_state import SqliteAuthState


def test_in_memory_client_public_keys() -> None:
    """Test client public keys store and get from state."""
    key_pairs = [generate_key_pairs() for _ in range(3)]
    public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

    in_memory_auth_state = InMemoryAuthState()
    in_memory_auth_state.store_client_public_keys(public_keys)

    assert in_memory_auth_state.get_client_public_keys() == public_keys


def test_sqlite_client_public_keys() -> None:
    """Test client public keys store and get from state."""
    key_pairs = [generate_key_pairs() for _ in range(3)]
    public_keys = {public_key_to_bytes(pair[1]) for pair in key_pairs}

    sqlite_auth_state = SqliteAuthState(":memory:")
    sqlite_auth_state.initialize()
    sqlite_auth_state.store_client_public_keys(public_keys)

    assert sqlite_auth_state.get_client_public_keys() == public_keys


def test_in_memory_node_id_public_key_pair() -> None:
    """Test store and get node_id public_key pair."""
    in_memory_auth_state = InMemoryAuthState()
    node_id = in_memory_auth_state.create_node()
    public_key = public_key_to_bytes(generate_key_pairs()[1])

    in_memory_auth_state.store_node_id_public_key_pair(node_id, public_key)

    assert in_memory_auth_state.get_public_key_from_node_id(node_id) == public_key


def test_sqlite_node_id_public_key_pair() -> None:
    """Test store and get node_id public_key pair."""
    sqlite_auth_state = SqliteAuthState(":memory:")
    sqlite_auth_state.initialize()
    node_id = sqlite_auth_state.create_node()
    public_key = public_key_to_bytes(generate_key_pairs()[1])

    sqlite_auth_state.store_node_id_public_key_pair(node_id, public_key)

    assert sqlite_auth_state.get_public_key_from_node_id(node_id) == public_key


def test_generate_shared_key() -> None:
    """Test util function generate_shared_key."""
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()

    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])

    assert client_shared_secret == server_shared_secret


def test_hmac() -> None:
    """Test util function compute and verify hmac."""
    client_keys = generate_key_pairs()
    server_keys = generate_key_pairs()
    client_shared_secret = generate_shared_key(client_keys[0], server_keys[1])
    server_shared_secret = generate_shared_key(server_keys[0], client_keys[1])
    message = b"Flower is the future of AI"

    client_compute_hmac = compute_hmac(client_shared_secret, message)

    assert verify_hmac(server_shared_secret, message, client_compute_hmac)