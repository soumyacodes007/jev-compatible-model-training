"""Dataset sources. Each returns ``{split_name: rows}`` for a dataset spec.

Rows may be a ``datasets.Dataset`` or any list of dicts. Register your own
source with ``jevkit.registry.sources.register("name")``.
"""
