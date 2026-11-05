# Revision execution

See the root README and `server/README.md` for the protocol, environments and
commands. The corrected Task 8(c) proposal is approved. Completed server runs
continue with `python -m revision.finalize --run` in the SDV environment; this
uses existing private caches and never schedules generator training. Aggregate
tables and figures can then be regenerated with `03_generate_outputs.py`.
