"""Compatibility import for existing ARX model adapters."""
from arx_client.rpc import ArxROS2RPCClient, NUM_ARM_JOINTS, NUM_POSE_DIMS
__all__ = ["ArxROS2RPCClient", "NUM_ARM_JOINTS", "NUM_POSE_DIMS"]
