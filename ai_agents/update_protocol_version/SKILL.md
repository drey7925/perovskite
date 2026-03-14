---
name: update_protocol_version
description: Updates the protocol version and protocol negotiation logic. Use this when the user asks for tasks related to network protocol versioning (e.g. adding versions, cleaning up old versioning code, etc).
---

perovskite_core/proto/game_rpc.proto contains a changelog of protocol changes near `min_protocol_version`.

Ask the user to list the protocol features being added in the new version. Look at the relevant code, and summarize the protocol features in a new entry in the changelog. Take note of the version number of the new protocol.

## Description of negotiation mechanism

perovskite_client/src/net_client/mod.rs contains MIN_PROTOCOL_VERSION and MAX_PROTOCOL_VERSION, which represent the minimum and maximum protocol versions that the client can handle.

perovskite_server/src/network_server/grpc_service.rs contains SERVER_MIN_PROTOCOL_VERSION and SERVER_MAX_PROTOCOL_VERSION, which represent the minimum and maximum protocol versions that the server can send to the client.

When a client logs into a server, the highest protocol version shared between the client and the server is selected. The server knows this version (it's in `effective_protocol_version` in perovskite_server::src::network_server::client_context::SharedContext).

In the network code in perovskite_core/src/network_server/client_context.rs, the server must use the `effective_protocol_version` to determine which protocol features to use when sending messages to the client. For example, if a new type of message is added in some protocol version,
servers should refrain from sending that message when the effective protocol version is too low.

## When removing support for an old protocol version:

perovskite_core/src/network_server/client_context.rs: Logic that can no longer be triggered by old protocol versions below SERVER_MIN_PROTOCOL_VERSION should be removed, as it is effectively dead code. Logic that is unconditionally triggered by protocol versions >= SERVER_MIN_PROTOCOL_VERSION should be made unconditional.

In all cases, leave the `if context.effective_protocol_version != SERVER_MAX_PROTOCOL_VERSION` check and "Your client is out of date" message in place, as they apply generally as a reminder to players to update their clients.

## When adding support for a new protocol version:

perovskite_core/src/network_server/client_context.rs: Any behavior that is specific to the new version should be guarded by checks against `effective_protocol_version`. Clients that do not support the new version should get behavior that depends on the type of feature:

* New messages, fields, etc sent from server to client: Don't send them. Leave protobuf submessages as Optional, scalar values as their defaults.
* New messages, fields, etc sent from client to server: Expect them to be missing. Either the feature triggered by these messages should be disabled, or a default value should be inferred.
    * For fields or messages that are always "necessary" in the new protocol version (e.g. client backpressure, RTT measurements, player position, etc) impute a reasonable default value that leaves behavior as close to unchanged as possible.
    * For fields or messages that happen in response to client actions unavailable in the old protocol version, simply do not invoke the feature because the client will not send the message.
