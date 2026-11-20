"""Bounded public-data retrieval and explainable contract-overlap screening."""

import re
from collections import Counter
from datetime import date, datetime, timezone
from itertools import combinations
from typing import Annotated, Any

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from mcp_server_n2ia.config import settings
from mcp_server_n2ia.services.usaspending import search_contract_awards

AGENCIES = [
    "General Services Administration",
    "Department of Homeland Security",
    "Department of the Treasury",
]
STOPWORDS = set(
    "the and for with this that from services service contract support provide "
    "base month months plus year years option options period task order issued "
    "providing required requirement requirements task order award modification "
    "under shall will are of to in on a an is as by at be or it".split()
)
LIMITATION = (
    "Screening candidates only: similar text and overlapping dates do not prove "
    "duplicate services, waste, consolidation feasibility, or savings. "
    "Review scopes, locations, task orders and parent vehicles manually. "
    "No SAM.gov or GAO enrichment is included."
)


class Award(BaseModel):
    """Only source-identified awards can be compared reproducibly."""

    source_award_key: str = Field(min_length=1)
    award_id: str | None = None
    awarding_agency: str | None = None
    contractor: str | None = None
    recipient_uei: str | None = None
    description: str | None = None
    psc: str | None = None
    naics: str | None = None
    performance_start_date: str | None = None
    performance_end_date: str | None = None
    source_url: str | None = None
    service_category: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    context_status: str = "not_requested"
    context_source: str | None = None
    parent_award_key: str | None = None
    parent_piid: str | None = None
    parent_agency: str | None = None
    parent_vehicle_type: str | None = None
    parent_source_url: str | None = None


def tokens(text: str | None) -> set[str]:
    """Use transparent word matching; no external embedding service required."""
    return {
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in STOPWORDS
    }


def interval(award: Award) -> tuple[date, date] | None:
    """Do not infer dates: missing or reversed performance periods are excluded."""
    try:
        start = date.fromisoformat(award.performance_start_date or "")
        end = date.fromisoformat(award.performance_end_date or "")
        return (start, end) if end >= start else None
    except ValueError:
        return None

def detect_overlap(
    awards: Annotated[list[Award], Field(max_length=600)],
    min_similarity: Annotated[float, Field(ge=0, le=1)] = settings.lexical_similarity_threshold,
    cross_agency_only: bool = True,
    max_candidates: Annotated[int, Field(ge=1, le=100)] = 10,
    analysis_start: date | None = None,
    analysis_end: date | None = None,
) -> dict[str, Any]:
    """Screen supplied awards using TF-IDF similarity and date overlap.

    Lexical matching requires:
    - TF-IDF cosine similarity >= min_similarity (default 0.20).
    - At least 2 shared informative words.

    Uses the existing production tokenization and STOPWORDS.
    Classification matches are supporting evidence, not proof of overlap.
    Counts describe only the supplied sample.
    """

    # Deduplicate awards using their global source keys.
    unique = {a.source_award_key: a for a in awards}

    # Preserve existing contract eligibility requirements.
    valid = [
        a for a in unique.values()
        if interval(a) and len(tokens(a.description)) >= 3
    ]

    candidates = []

    # Build TF-IDF vectors once for all valid contracts.
    if valid:
        vectorizer = TfidfVectorizer(
            analyzer=lambda text: sorted(tokens(text)),
            lowercase=False,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )

        tfidf_matrix = vectorizer.fit_transform(
            [a.description or "" for a in valid]
        )

        # Index vectors by their position in the valid-awards list.
        token_sets = [tokens(a.description) for a in valid]
        intervals = [interval(a) for a in valid]

        for i, j in combinations(range(len(valid)), 2):
            left, right = valid[i], valid[j]

            # Preserve agency filtering.
            if not left.awarding_agency or not right.awarding_agency:
                continue

            if (
                cross_agency_only
                and left.awarding_agency == right.awarding_agency
            ):
                continue

            # Preserve inclusive performance-period filtering.
            lstart, lend = intervals[i]
            rstart, rend = intervals[j]

            start = max(
                lstart,
                rstart,
                *([analysis_start] if analysis_start else [])
            )
            end = min(
                lend,
                rend,
                *([analysis_end] if analysis_end else [])
            )

            if start > end:
                continue

            # Hybrid lexical matching.
            shared = token_sets[i] & token_sets[j]

            # Two shared informative words are required.
            if len(shared) < settings.min_shared_informative_words:
                continue

            # TF-IDF cosine similarity.
            similarity = float(
                tfidf_matrix[i].multiply(tfidf_matrix[j]).sum()
            )

            if similarity < min_similarity:
                continue

            # Classification matches provide supporting evidence.
            matching = [
                field
                for field in ("psc", "naics")
                if getattr(left, field)
                and getattr(left, field) == getattr(right, field)
            ]

            # Shared parent vehicle is context, not proof of duplication.
            same_parent = bool(
                left.parent_award_key
                and left.parent_award_key == right.parent_award_key
            )

            parent_note = (
                "Shared parent vehicle: review distinct task-order scopes."
                if same_parent
                else "Parent relationship unknown."
                if (
                    left.context_status != "retrieved"
                    or right.context_status != "retrieved"
                )
                else "No shared parent vehicle reported."
            )

            candidates.append(
                {
                    "left": left.model_dump(),
                    "right": right.model_dump(),
                    "text_similarity": round(similarity, 4),
                    "shared_parent_vehicle": same_parent,
                    "parent_context_note": parent_note,
                    "overlap_start": str(start),
                    "overlap_end": str(end),
                    "overlap_days": (end - start).days + 1,
                    "shared_terms": sorted(shared),
                    "matching_classifications": matching,
                    "explanation": (
                        f"TF-IDF similarity {similarity:.4f}; "
                        f"{len(shared)} shared informative words; "
                        f"overlapping performance dates; "
                        f"matching classifications: "
                        f"{', '.join(matching) or 'none'}."
                    ),
                }
            )

    # Rank candidates by TF-IDF similarity.
    candidates.sort(
        key=lambda pair: (
            -pair["text_similarity"],
            pair["left"]["source_award_key"],
            pair["right"]["source_award_key"],
        )
    )

    # Preserve existing sample fragmentation analysis.
    groups: dict[str, list[Award]] = {}

    for award in unique.values():
        groups.setdefault(award.psc or "unknown", []).append(award)

    fragmentation = []

    for psc, group in sorted(groups.items()):
        vendors = {
            a.recipient_uei
            or (a.contractor or "").strip().casefold()
            for a in group
        } - {""}

        fragmentation.append(
            {
                "psc": psc,
                "award_count": len(group),
                "identified_vendor_count": len(vendors),
                "agency_counts": dict(
                    Counter(
                        a.awarding_agency or "unknown"
                        for a in group
                    )
                ),
            }
        )

    return {
        "input_count": len(awards),
        "unique_awards": len(unique),
        "excluded_missing_dates_or_description": (
            len(unique) - len(valid)
        ),
        "candidate_count": len(candidates),
        "candidates": candidates[:max_candidates],
        "candidates_truncated": len(candidates) > max_candidates,
        "sample_fragmentation": fragmentation,
        "limitations": LIMITATION,
    }


async def retrieve_contracts(
    start_date: date,
    end_date: date,
    agencies: Annotated[list[str], Field(min_length=1, max_length=3)] = AGENCIES,
    keyword: str = "cloud",
    pages_per_agency: Annotated[int, Field(ge=1, le=2)] = 1,
    page_size: Annotated[int, Field(ge=1, le=100)] = 50,
    psc_codes: list[str] | None = None,
    naics_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve up to 600 awards with per-agency coverage and partial failures.

    Activity dates filter award selection, not performance periods. Defaults
    cover GSA, DHS and Treasury. Every page is a bounded read-only API request.
    complete means pagination finished for these filters, not all IT contracts.
    """
    # The key reflects the exact API query, including ordering and pagination.
    # Search snapshots retain exact award records for reproducible evaluations.
    from mcp_server_n2ia.services import contract_cache
    import os
    cache_mode = os.getenv('N2IA_CONTRACT_CACHE_MODE', 'auto').lower()
    if cache_mode not in ('auto', 'live', 'cached'):
        return {'error': 'invalid_cache_mode', 'hint': 'Use auto, live or cached.'}
    query_key = {
        'start_date': str(start_date), 'end_date': str(end_date),
        'agencies': list(dict.fromkeys(a.strip() for a in agencies)),
        'keyword': keyword, 'pages_per_agency': pages_per_agency,
        'page_size': page_size, 'psc_codes': sorted(psc_codes or []),
        'naics_codes': sorted(naics_codes or []),
        'sort': 'Award Amount', 'order': 'desc', 'api_version': 'v2',
    }
    if cache_mode == 'cached' and not contract_cache.enabled():
        return {'error': 'cache_unavailable', 'hint': 'SQL warehouse is not configured.'}
    if cache_mode != 'live' and contract_cache.enabled():
        try:
            cached = contract_cache.load_search(query_key, allow_expired=(cache_mode == 'cached'))
            if cached is not None:
                return cached
        except Exception as exc:
            if cache_mode == 'cached':
                return {'error': 'cache_read_failed', 'hint': str(exc)}
            # auto falls back to the live API; report cache failure below.
            cache_read_error = str(exc)
        else:
            cache_read_error = None
        if cache_mode == 'cached':
            return {'error': 'cache_miss', 'hint': 'No matching saved search snapshot.'}
    else:
        cache_read_error = None
    if end_date < start_date:
        return {"error": "invalid_dates", "hint": "End must follow start."}
    items, coverage, seen = [], [], set()
    for agency in dict.fromkeys(agencies):
        agency = agency.strip()
        if not agency:
            return {"error": "invalid_agency", "hint": "Use official agency names."}
        entry = {"agency": agency, "pages_received": 0, "complete": False}
        for page in range(1, pages_per_agency + 1):
            result = await search_contract_awards(
                start_date,
                end_date,
                agency=agency,
                keyword=keyword,
                page=page,
                limit=page_size,
                psc_codes=psc_codes,
                naics_codes=naics_codes,
            )
            if "error" in result:
                entry["error"] = result
                break
            entry["pages_received"] += 1
            entry["has_more"] = result["has_more"]
            for row in result["items"]:
                key = row["source_award_key"]
                # Unidentified rows cannot safely be deduplicated or compared.
                if key and key not in seen:
                    seen.add(key)
                    items.append(row)
            if not result["has_more"]:
                entry["complete"] = True
                break
        coverage.append(entry)
    result = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "count": len(items),
        "coverage": coverage,
        "complete": all(c["complete"] for c in coverage),
        "query": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "keyword": keyword,
            "psc_codes": psc_codes,
            "naics_codes": naics_codes,
        },
        "limitations": "Bounded sample unless all coverage entries complete. "
        "Award amounts are not interval spending or savings.",
    }
    if contract_cache.enabled():
        try:
            result['cache'] = contract_cache.save_search(query_key, result)
        except Exception as exc:
            # Live analysis still works, but persistence failure is explicit.
            result['cache'] = {'hit': False, 'persisted': False, 'error': str(exc)}
    else:
        result['cache'] = {'hit': False, 'persisted': False,
                           'error': 'N2IA_SQL_WAREHOUSE_ID not configured'}
    if cache_read_error:
        result['cache']['read_error'] = cache_read_error
    return result



# Broad topic words are not evidence that two contracts perform the same work.
# Kept separate from STOPWORDS so existing lexical/semantic diagnostics stay unchanged.
GENERIC_GROUP_TERMS = set(
    "cybersecurity cyber security information technical technology agency "
    "department federal government program programs management managed "
    "operations operational engineering professional enterprise systems system "
    "solution solutions activities assistance expertise development infrastructure "
    "center office division services service support".split()
)


# Explicit service functions, not broad keywords such as "cybersecurity".
# Multi-label classification is deliberate: some awards describe several functions.
SERVICE_FUNCTION_RULES = {
    "risk_management_compliance": (
        r"\brisk management framework\b|\brmf\b|\bcompliance\b|"
        r"\bgovernance\b|\bauthorization to operate\b|\bato\b"
    ),
    "vulnerability_management": (
        r"\bvulnerability management\b|\bvulnerability assessment(?:s)?\b|"
        r"\bpenetration test(?:ing)?\b|\bvm cybersecurity\b"
    ),
    "security_operations": (
        r"\bsecurity operations cent(?:er|re)\b|\bsoc\b|"
        r"\bcybersecurity operations\b|\bincident response\b|"
        r"\bthreat monitoring\b"
    ),
    "architecture_engineering": (
        r"\benterprise architectur(?:e|al)\b|\bsecurity architectur(?:e|al)\b|"
        r"\bcybersecurity architectur(?:e|al)\b|\bsystems engineering\b|"
        r"\bcybersecurity engineering\b|"
        r"\bcybersecurity operations and engineering\b"
    ),
    "assessments_audits": (
        r"\bcybersecurity assessments?\b|\bsecurity assessments?\b|"
        r"\bsecurity audits?\b|\bcybersecurity audits?\b"
    ),
    "privacy_records": (
        r"\bprivacy\b|\brecords management\b|\binformation management\b"
    ),
    "acquisition_program_support": (
        r"\bprocurement management\b|\bacquisition support\b|"
        r"\bcontract management\b|\bprogram management\b"
    ),

    "cybersecurity_sme_support": (
        r"\bcybersecurity\b.*\b(?:sme|subject matter expert)\b"
    ),
    "counter_ied": r"\bcounter.improvised explosive device\b|\bcounter.ied\b",
}


def classify_service_functions(description: str | None) -> list[str]:
    """Conservative, reproducible multi-label description classification."""
    text = re.sub(r"\s+", " ", (description or "").lower())
    return [name for name, pattern in SERVICE_FUNCTION_RULES.items()
            if re.search(pattern, text)]


def group_overlap_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_groups: int = 10,
    min_semantic_similarity: float = settings.semantic_similarity_threshold,
) -> dict[str, Any]:
    """Create evidence-screened relationships and split broad components into communities.

    Cross-agency matching is preserved. A community is a review aid, not a claim
    that every member matches every other member.
    """
    import networkx as nx

    graph = nx.Graph()
    screened_out = 0
    screened_out_function = 0
    for pair in candidates:
        specific_terms = sorted(set(pair.get("shared_terms", [])) - GENERIC_GROUP_TERMS)
        classification_match = pair["matching_psc"] or pair["matching_naics"]
        semantic_strong = pair["semantic_similarity"] >= min_semantic_similarity
        lexical_strong = pair["match_method"] in ("both", "lexical_only")

        # Allow either a supported semantic match or a supported lexical match.
        semantic_supported = semantic_strong and (classification_match or len(specific_terms) >= settings.min_shared_specific_terms_for_grouping)
        lexical_supported = lexical_strong and (len(specific_terms) >= settings.min_shared_specific_terms_for_grouping or semantic_strong)
        if not (semantic_supported or lexical_supported):        
            screened_out += 1
            continue
        left, right = pair["left"], pair["right"]
        left_functions = classify_service_functions(left.get("description"))
        right_functions = classify_service_functions(right.get("description"))
        common_functions = sorted(set(left_functions) & set(right_functions))
        # Require an explicitly shared function; classification codes alone
        # are not evidence that the contracts perform the same kind of work.
        if not common_functions:
            screened_out += 1
            screened_out_function += 1
            continue
        lk, rk = left["source_award_key"], right["source_award_key"]
        graph.add_node(lk, **left)
        graph.add_node(rk, **right)
        graph.add_edge(lk, rk, **{
            **{k: v for k, v in pair.items() if k not in ("left", "right")},
            "specific_shared_terms": specific_terms,
            "shared_service_functions": common_functions,
            "weight": pair["semantic_similarity"],
        })

    communities = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        # Modularity communities avoid treating a long chain as one service area.
        # Only split components of 4+; tiny groups remain understandable.
        if len(component) >= 4 and subgraph.number_of_edges() > 1:
            partitions = nx.community.greedy_modularity_communities(subgraph, weight="weight")
        else:
            partitions = [set(component)]
        for partition in partitions:
            if len(partition) >= 2:
                communities.append(graph.subgraph(partition).copy())

    groups = []
    for subgraph in communities:
        contracts = [
            {**dict(subgraph.nodes[key]),
             "service_functions": classify_service_functions(subgraph.nodes[key].get("description"))}
            for key in sorted(subgraph.nodes)
        ]
        relationships = []
        for lk, rk, details in subgraph.edges(data=True):
            relationships.append({
                "award_key_a": lk, "award_key_b": rk,
                "award_id_a": subgraph.nodes[lk].get("award_id"),
                "award_id_b": subgraph.nodes[rk].get("award_id"),
                **{k: v for k, v in details.items() if k != "weight"},
            })
        relationships.sort(key=lambda e: (
            -e["semantic_similarity"], -int(e["matching_psc"]),
            -int(e["matching_naics"]), e["award_key_a"], e["award_key_b"],
        ))
        agencies = sorted({c["awarding_agency"] for c in contracts if c.get("awarding_agency")})
        functions = sorted({f for e in relationships for f in e["shared_service_functions"]})
        groups.append({
            "group_id": "",
            "contract_count": len(contracts),
            "agency_count": len(agencies),
            "agencies": agencies,
            "service_functions": functions,
            "relationship_count": len(relationships),
            "possible_relationship_count": len(contracts) * (len(contracts) - 1) // 2,
            "connection_density": round(nx.density(subgraph), 3),
            "strongest_semantic_similarity": max(e["semantic_similarity"] for e in relationships),
            "average_semantic_similarity_on_edges": round(
                sum(e["semantic_similarity"] for e in relationships) / len(relationships), 4
            ),
            "contracts": contracts,
            "relationships": relationships,
            "interpretation": (
                "A community of contracts connected by screened semantic and supporting "
                "classification or specific-word evidence; not all members necessarily match. "
                "Requires acquisition review."
            ),
        })
    groups.sort(key=lambda g: (
        -g["relationship_count"], -g["connection_density"],
        -g["average_semantic_similarity_on_edges"],
        -g["contract_count"], g["contracts"][0]["source_award_key"],
    ))
    for i, group in enumerate(groups, 1):
        group["group_id"] = f"G{i}"
    return {
        "group_count": len(groups),
        "groups": groups[:max_groups],
        "groups_truncated": len(groups) > max_groups,
        "grouping_method": "Screened lexical or semantic evidence + shared service functions + NetworkX communities",
        "grouping_screened_out_relationships": screened_out,
        "grouping_screened_out_function_mismatch": screened_out_function,
        "service_function_method": "Conservative rule-based multi-label classification from award descriptions",
        "grouping_relationship_count": graph.number_of_edges(),
        "grouping_limitation": (
            "Function rules are incomplete and short descriptions may omit relevant scope. "
            "Only listed relationships passed screening; community membership does not "
            "imply all-to-all similarity or proven duplicate spending. "
            "Single contracts without a qualifying relationship are not shown in groups."
        ),
    }


def compare_awards(
    awards: list[Award],
    *,
    min_lexical_similarity: float = settings.lexical_similarity_threshold,
    min_semantic_similarity: float = settings.semantic_similarity_threshold,
    cross_agency_only: bool = True,
    max_examples_per_category: int = 10,
    analysis_start: date | None = None,
    analysis_end: date | None = None,
) -> dict[str, Any]:
    """Compare awards, restricting performance overlap to the requested window when provided."""
    unique = {award.source_award_key: award for award in awards}
    valid = [
        award for award in unique.values()
        if interval(award) is not None
        and len(tokens(award.description)) >= 3
    ]
    categories = {name: [] for name in ("both", "lexical_only", "semantic_only")}
    counts = {name: 0 for name in ("both", "lexical_only", "semantic_only", "neither")}
    eligible_pairs = 0
    ranked_candidates = []
    aligned_classification_pairs = 0
    excluded_same_agency = 0
    excluded_missing_agency = 0
    excluded_nonoverlapping_dates = 0

    if len(valid) >= 2:
        from mcp_server_n2ia.services.semantic_similarity import calculate_embeddings
        embeddings = calculate_embeddings([a.description for a in valid])

        # TF-IDF vectors using existing production tokenization.
        vectorizer = TfidfVectorizer(
            analyzer=lambda text: sorted(tokens(text)),
            lowercase=False,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )

        tfidf_matrix = vectorizer.fit_transform(
            [award.description or "" for award in valid]
        )

        token_sets = [tokens(a.description) for a in valid]
        intervals = [interval(a) for a in valid]

        for i, j in combinations(range(len(valid)), 2):
            left, right = valid[i], valid[j]
            if not left.awarding_agency or not right.awarding_agency:
                excluded_missing_agency += 1
                continue
            if cross_agency_only and left.awarding_agency == right.awarding_agency:
                excluded_same_agency += 1
                continue
            left_start, left_end = intervals[i]
            right_start, right_end = intervals[j]
            start = max(left_start, right_start, *([analysis_start] if analysis_start else []))
            end = min(left_end, right_end, *([analysis_end] if analysis_end else []))
            if start > end:
                excluded_nonoverlapping_dates += 1
                continue
            eligible_pairs += 1
            matching_psc = bool(left.psc and right.psc and left.psc == right.psc)
            matching_naics = bool(left.naics and right.naics and left.naics == right.naics)
            if matching_psc or matching_naics:
                aligned_classification_pairs += 1
            shared = token_sets[i] & token_sets[j]
            lexical_score = float(tfidf_matrix[i].multiply(tfidf_matrix[j]).sum())
            semantic_score = float(embeddings[i] @ embeddings[j])
            lexical_match = len(shared) >= settings.min_shared_informative_words and lexical_score >= min_lexical_similarity
            semantic_match = semantic_score >= min_semantic_similarity
            category = (
                "both" if lexical_match and semantic_match else
                "lexical_only" if lexical_match else
                "semantic_only" if semantic_match else "neither"
            )
            counts[category] += 1
            if category == "neither":
                continue
            reasons = []
            if not lexical_match:
                if len(shared) < settings.min_shared_informative_words:
                    reasons.append(f"fewer_than_{settings.min_shared_informative_words}_shared_informative_words")
                if lexical_score < min_lexical_similarity:
                    reasons.append("lexical_score_below_threshold")
            example = {
                "left": left.model_dump(mode="json"),
                "right": right.model_dump(mode="json"),
                "lexical_similarity": round(lexical_score, 4),
                "semantic_similarity": round(semantic_score, 4),
                "shared_terms": sorted(shared),
                "overlap_start": start.isoformat(),
                "overlap_end": end.isoformat(),
                "overlap_days": (end - start).days + 1,
                "lexical_rejection_reasons": reasons,
                "matching_psc": matching_psc,
                "matching_naics": matching_naics,
                "classification_alignment": (
                    "psc_and_naics" if matching_psc and matching_naics else
                    "psc" if matching_psc else "naics" if matching_naics else "none"
                ),
            }
            categories[category].append(example)
            # Rank first by PSC/NAICS alignment, then semantic evidence.
            # Keep all qualifying pairs, including different-classification pairs.
            example["match_method"] = category
            example["review_reason"] = (
                "Same PSC and NAICS; descriptions suggest related work."
                if matching_psc and matching_naics else
                "Same PSC; descriptions suggest related work."
                if matching_psc else
                "Same NAICS; descriptions suggest related work."
                if matching_naics else
                "Descriptions suggest related work despite different or missing classifications."
            )
            ranked_candidates.append(example)

    grouped = group_overlap_candidates(
        ranked_candidates, max_groups=max_examples_per_category,
        min_semantic_similarity=min_semantic_similarity
    )

    # Classification is a prioritization signal, not a hard exclusion.
    ranked_candidates.sort(
        key=lambda x: (
            -int(x["matching_psc"]),
            -int(x["matching_naics"]),
            -x["semantic_similarity"],
            -x["lexical_similarity"],
            x["left"]["source_award_key"],
            x["right"]["source_award_key"],
        )
    )
    # Rank and limit the returned evidence, but never truncate the counts.
    for category, examples in categories.items():
        score_key = "lexical_similarity" if category == "lexical_only" else "semantic_similarity"
        examples.sort(
            key=lambda x: (-x[score_key], x["left"]["source_award_key"], x["right"]["source_award_key"])
        )
        categories[category] = examples[:max_examples_per_category]

    return {
        "input_count": len(awards),
        "unique_awards": len(unique),
        "analyzed_awards": len(valid),
        "excluded_missing_dates_or_description": len(unique) - len(valid),
        "eligible_pairs_compared": eligible_pairs,
        "classification_aligned_pairs": aligned_classification_pairs,
        "ranked_candidates": ranked_candidates[:max_examples_per_category],
        **grouped,
        "ranked_candidate_count": len(ranked_candidates),
        "ranked_candidates_truncated": len(ranked_candidates) > max_examples_per_category,
        "ranking_method": "PSC match, then NAICS match, then semantic and lexical similarity; screening only",
        "excluded_pairs": {
            "missing_agency": excluded_missing_agency,
            "same_agency": excluded_same_agency,
            "nonoverlapping_performance_dates": excluded_nonoverlapping_dates,
        },
        "thresholds": {
            "min_lexical_similarity": min_lexical_similarity,
            "min_shared_informative_words": settings.min_shared_informative_words,
            "min_semantic_similarity": min_semantic_similarity,
        },
        "counts": {
            **counts,
            "lexical_total": counts["lexical_only"] + counts["both"],
            "semantic_total": counts["semantic_only"] + counts["both"],
        },
        "examples": categories,
        "examples_truncated": {
            name: counts[name] > len(categories[name]) for name in categories
        },
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "limitations": LIMITATION,
    }