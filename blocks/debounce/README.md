Debounce
========
The Debounce block will filter out signals for **interval** seconds after a signal flows through the block.

Properties
----------
- **group_by**: The signal attribute on the incoming signal whose values will be used to define groups on the outgoing signal.
- **interval**: Amount of time to wait before emitting another signal from the same group.

Inputs
------
- **default**: Any list of signals.

Outputs
-------
- **default**: At every interval, the first signal in each group.

Commands
--------
- **groups**: Display the existing groups.

Dependencies
------------
None

