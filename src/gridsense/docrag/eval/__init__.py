"""Eval harness for DocRAG: the golden dataset, RAGAS scoring, and the merge gate.

:mod:`golden` loads the labelled set and computes deterministic retrieval recall,
:mod:`ragas_eval` runs the real chain over it and scores the answers, and
:mod:`thresholds` holds the gate that CI fails on.
"""
