"""Deterministic, answer-blind evidence retrieval from the existing KG adapters.

Only the visible source name and requested node types are used for traversal.
In particular, a two-hop query never uses BioHopR's hidden hop1 entity name.
"""
import json

from src.kg_eval.utils import normalize


QUERY_FIELDS = ("hop", "question", "source_name", "source_type", "intermediate_type", "target_type")


def retrieval_key(question):
    """Cache identical retrieval requests without incorporating reference answers."""
    return tuple(question.get(field, "") for field in QUERY_FIELDS)


def _node_text(graph, node_id):
    # JSON quoting keeps a source label containing a newline on one evidence line.
    name = json.dumps(graph.names[node_id] or "[unlabelled]", ensure_ascii=False)
    native_id = json.dumps(graph.external_ids[node_id], ensure_ascii=False)
    return f"{name} [{graph.types[node_id]}; ID={native_id}]"


def _path_text(graph, path):
    # Adapters retain positive semantic adjacency but pool other relation labels.
    # A generic association never fabricates a more specific source predicate.
    return " -- associated with -- ".join(_node_text(graph, node) for node in path)


def retrieve_context(question, graph, max_evidence_paths=24, relation_policy="semantic", *, supported_types=None):
    """Return a bounded set of real graph paths, independent of reference answers.

    Count every reachable endpoint ID. To avoid filling the prompt with repeated
    endpoints, keep one real path per endpoint, taking the first path in native-ID
    order, then cap endpoints in native-ID order. This is deterministic and has
    no learned or reference-informed ranking. ``evidence_truncated`` indicates
    omitted endpoint representatives; alternate paths to a retained endpoint are
    intentionally deduplicated before applying the cap.
    """
    if max_evidence_paths <= 0:
        raise ValueError("max_evidence_paths must be positive")
    if relation_policy not in {"semantic", "type-only"}:
        raise ValueError("Unknown relation policy")
    hop = question.get("hop")
    if hop not in (1, 2):
        raise ValueError("Question hop must be 1 or 2")
    result = {"question_id": question["question_id"], "status": "ok", "evidence": [],
              "n_candidates": 0, "n_start_matches": 0, "evidence_truncated": False}
    source = normalize(question.get("source_name"))
    source_type = normalize(question.get("source_type"))
    target_type = normalize(question.get("target_type"))
    intermediate = normalize(question.get("intermediate_type")) if hop == 2 else ""
    type_path = [source_type] + ([intermediate] if hop == 2 else []) + [target_type]
    # Do not inspect question['targets'], ['applicable'] or ['exclusion_reason']:
    # even whether a row has reference answers must not alter graph retrieval.
    if not normalize(question.get("question")):
        result["status"] = "missing_question"
        return result
    if not source or not all(type_path):
        result["status"] = "missing_fields"
        return result
    supported_types = graph.supported_types if supported_types is None else supported_types
    if not all(node_type in supported_types for node_type in type_path):
        result["status"] = "unsupported_type"
        return result
    if not graph.supports_path(type_path, relation_policy):
        result["status"] = "unsupported_relation"
        return result
    starts = graph.resolve(source, source_type)
    result["n_start_matches"] = len(starts)
    if not starts:
        result["status"] = "start_not_found"
        return result

    native_id = lambda node: graph.external_ids[node]
    representative_paths = {}
    for start in sorted(starts, key=native_id):
        if hop == 1:
            for end in graph.neighbor_ids({start}, target_type, relation_policy):
                representative_paths.setdefault(end, (start, end))
        else:
            for bridge in sorted(graph.neighbor_ids({start}, intermediate, relation_policy), key=native_id):
                for end in graph.neighbor_ids({bridge}, target_type, relation_policy):
                    representative_paths.setdefault(end, (start, bridge, end))
    result["n_candidates"] = len(representative_paths)
    if not representative_paths:
        result["status"] = "no_candidates"
        return result
    selected = sorted(representative_paths, key=native_id)[:max_evidence_paths]
    result["evidence"] = [_path_text(graph, representative_paths[end]) for end in selected]
    result["evidence_truncated"] = len(representative_paths) > max_evidence_paths
    return result
