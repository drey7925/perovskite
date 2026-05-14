# Cart routing system design

This is the high-level design of cart routing. It is intended to complement DESIGN_MOVEMENT.md - that doc focuses
on lower-level cart movement and operations, while the document focuses on a higher-level routing system.

This is a greenfield design - this doc will be filled in as development proceeds.


## Important factors to keep in mind

* Carts move at up to 90 m/s. This is fast in world-coordinate terms, but slow in terms of computational time. A
  well-written pathfinding operation can easily run faster than a cart, all while warming the cache for that cart.
* Individual block reads (including with extended data) are cheap.
* Immediate consistency is neither achievable at scale nor desired (some degree of eventual consistency and discovery delay is acceptable and expected as pat of gameplay)

## Waypoints

Waypoints are the fundamental building block of the cart routing system.

A waypoint is a location that's relevant for cart routing/dispatch in some way. It has a location (block coordinate)
and a direction (CompassDirection) - bidirectional tracks can have two waypoints near each other or potentially at the same location
(note that right now, there's no way to put two waypoints at the same location, since all of the existing waypoint-bearing blocks are unidirectional, but that may change!). The coordiante and direction together uniquely identify a waypoint.

In graph-based pathfinding, waypoints are the nodes of the graph, and the edges are the segments of track between
waypoints. Waypoints will later be produced by blocks that are located at the Y+2 location where signals and
speedposts are currently found. Note that despite the marker for the waypoint being at Y+2 above the track, it is the track itself
that is the waypoint.

Examples of waypoints include:
* Interlocking entrances (first interlocking signal encountered when approaching an interlocking)
* Interlocking exits (first non-interlocking signal, or dead end, encountered when leaving an interlocking)
* Starting signals for interlocked platform stops
* Arbitrary waypoints placed by players (later on)
* Other signals that may arise as part of the routing design.

### Waypoint data

**Naming:** Occasionally, it may be useful for players to name a waypoint in some human-readable way (e.g. Spawnville Station, Platform 1, northbound). The naming format is TBD, but will likely be hierarchical to allow for hierarchical routing later on.

**Waypoint properties:** For routing functionality, waypoints will need to have distinct types; see the list of example types above. They will also have names, in a yet-to-be-determined naming scheme.

We have not yet committed to whether waypoint data is cached with a reference to a waypoint, or loaded on the fly
from the actual waypoint block each time. Presumably, it's just as easy to mess up a link between waypoints (by disturbing track) as it is
to change the waypoint itself, so it's probably fine to cache waypoint data along with coordinates when caching adjacencies.