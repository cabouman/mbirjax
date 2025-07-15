jax produces a subtle bug in which a given variable can have two different values depending on how the variable is
accessed.

In the attached code,

jax_rounding_bug.py illustrates this in a simple script.  The two displayed images should be the same but are not.

parallel_beam.py includes jax.debug_print statements that give different results depending on whether n[0:8] or
n[0:9] is printed.
