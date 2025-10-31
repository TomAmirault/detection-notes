# Maximum duration (in minutes) : maximum total duration of a group.
# If the group exceeds this duration (in minutes), the group is closed and a new one is started.
MAX_GROUP_DURATION = 2 # minutes

# Maximum pause (in secondes) : maximum allowed gap between two pieces of information for them to be considered part of the same group.
# If the pause between two logs exceeds this value, a new group is created.
MAX_PAUSE = 30 # secondes   