"""Unity Catalog Delta-backed cache for bounded USAspending search snapshots.

Requires a SQL warehouse accessible to the Databricks App service principal.
Set N2IA_SQL_WAREHOUSE_ID and optionally N2IA_CACHE_SCHEMA.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from uuid import uuid4

from databricks.sdk import WorkspaceClient

import logging

logger = logging.getLogger(__name__)

SCHEMA = os.getenv('N2IA_CACHE_SCHEMA', 'users.`satish_venkataraman`')
# SQL identifiers cannot be supplied as bound values; restrict the configured name.
import re
if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.`[A-Za-z_][A-Za-z0-9_.]*`', SCHEMA):
    raise ValueError('N2IA_CACHE_SCHEMA must look like users.`satish_venkataraman`')


def enabled():
    return bool(os.getenv('N2IA_SQL_WAREHOUSE_ID'))


@lru_cache(maxsize=1)
def client():
    return WorkspaceClient()


def sql_literal(value):
    """Quote values for SQL statements, including JSON with single quotes."""
    if value is None:
        return 'NULL'
    return "'" + str(value).replace("'", "''") + "'"


def run_sql(statement):
    warehouse_id = os.getenv('N2IA_SQL_WAREHOUSE_ID')
    if not warehouse_id:
        raise RuntimeError('N2IA_SQL_WAREHOUSE_ID is not configured')
    response = client().statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout='50s',
    )
    state = str(response.status.state) if response.status else 'UNKNOWN'
    if 'SUCCEEDED' not in state:
        raise RuntimeError(f'Databricks SQL statement did not succeed: {state}; '
                           f'{response.status.error if response.status else "no status"}')
    return (response.result.data_array or []) if response.result else []


def search_key(query):
    canonical = json.dumps(query, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def load_search(query, *, max_age_hours=24, allow_expired=False):
    key = search_key(query)
    logger.info("CACHE SEARCH: key=%s", key[:12])
    rows = run_sql(f"""
        SELECT search_id, retrieved_at_utc, expires_at_utc, contract_snapshot_json,
               contract_count, coverage, complete, limitations
        FROM {SCHEMA}.cache_contract_searches
        WHERE search_key = {sql_literal(key)}
        {' ' if allow_expired else 'AND expires_at_utc > current_timestamp()'}
        ORDER BY retrieved_at_utc DESC LIMIT 1
    """)
    if not rows:
        logger.info("CACHE MISS: key=%s", key[:12])
        return None
    search_id, retrieved, expires, snapshot, count, coverage, complete, limitations = rows[0]
    if not snapshot:
        raise ValueError(f'Cache snapshot {search_id} is missing contract data')
    items = json.loads(snapshot)
    if len(items) != int(count):
        raise ValueError(f'Cache snapshot {search_id} has inconsistent count')
    return {
        'items': items, 'count': len(items), 'coverage': json.loads(coverage),
        'complete': str(complete).lower() == 'true', 'limitations': limitations,
        'query': query, 'retrieved_at_utc': str(retrieved),
        'cache': {'hit': True, 'search_id': search_id, 'search_key': key,
                  'expires_at_utc': str(expires)},
    }


def save_search(query, retrieved, *, ttl_hours=24):
    """Persist immutable search snapshot, then update current award records.

    Snapshot is inserted first: failed award upserts never lose the fixed snapshot.
    """
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(hours=ttl_hours)
    search_id = str(uuid4())
    items = retrieved['items']
    logger.info(
        "CACHE WRITE START: contracts=%d",
        len(items),
    )
    payload = json.dumps(items, ensure_ascii=False, separators=(',', ':'), default=str)
    coverage = json.dumps(retrieved['coverage'], ensure_ascii=False, separators=(',', ':'))
    # Protect the SQL statement size; raise rather than silently omit rows.
    if len(payload.encode('utf-8')) > 3_000_000:
        raise ValueError('Search snapshot exceeds 3 MB; use smaller pages or bulk ingestion')
    keys = [item['source_award_key'] for item in items if item.get('source_award_key')]
    array_keys = 'array(' + ','.join(map(sql_literal, keys)) + ')' if keys else 'array()'
    run_sql(f"""
        INSERT INTO {SCHEMA}.cache_contract_searches
        (search_id, search_key, search_parameters, retrieved_at_utc, expires_at_utc,
         contract_keys, contract_snapshot_json, contract_count, coverage, complete, limitations)
        VALUES ({sql_literal(search_id)}, {sql_literal(search_key(query))},
                {sql_literal(json.dumps(query, sort_keys=True))},
                TIMESTAMP {sql_literal(now.isoformat())},
                TIMESTAMP {sql_literal(expiry.isoformat())},
                {array_keys}, {sql_literal(payload)}, {len(items)},
                {sql_literal(coverage)}, {str(bool(retrieved['complete'])).lower()},
                {sql_literal(retrieved.get('limitations', ''))})
    """)
    logger.info(
        "CACHE AWARDS SAVED: search_id=%s unique_contracts=%d",
        search_id,
        len(unique),
    )
    # Use small batches to stay below statement limits. Upserts are idempotent.
    columns = ['source_award_key', 'award_id', 'awarding_agency', 'contractor',
               'recipient_uei', 'description', 'psc', 'naics',
               'performance_start_date', 'performance_end_date',
               'award_obligations_usd', 'last_modified_date', 'source_url']
    unique = {row['source_award_key']: row for row in items if row.get('source_award_key')}
    for offset in range(0, len(unique), 20):
        batch = list(unique.values())[offset:offset + 20]
        values = []
        for row in batch:
            fields = []
            for col in columns:
                val = row.get(col)
                if col == 'award_obligations_usd':
                    fields.append('NULL' if val is None else f'TRY_CAST({sql_literal(val)} AS DOUBLE)')
                else:
                    fields.append(sql_literal(val))
            fields.append(f'TIMESTAMP {sql_literal(now.isoformat())}')
            values.append('(' + ','.join(fields) + ')')
        all_cols = columns + ['retrieved_at_utc']
        select_cols = ','.join(f'v.{c}' for c in all_cols)
        run_sql(f"""
            MERGE INTO {SCHEMA}.cache_contract_awards AS t
            USING (SELECT * FROM VALUES {','.join(values)} AS v({','.join(all_cols)})) AS s
            ON t.source_award_key = s.source_award_key
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
    return {'hit': False, 'search_id': search_id, 'search_key': search_key(query),
            'expires_at_utc': expiry.isoformat()}

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def model_version():
    # Explicit revision value should change when the model artifact changes.
    return os.getenv('N2IA_EMBEDDING_MODEL_VERSION', 'volume-initial')


def description_hash(description):
    return hashlib.sha256((description or '').encode('utf-8')).hexdigest()


def load_embeddings(descriptions):
    hashes = list(dict.fromkeys(description_hash(d) for d in descriptions))
    found = {}
    for offset in range(0, len(hashes), 40):
        part = hashes[offset:offset + 40]
        rows = run_sql(f"""
            SELECT description_hash, to_json(embedding)
            FROM {SCHEMA}.cache_contract_embeddings
            WHERE model_name = {sql_literal(MODEL_NAME)}
              AND model_version = {sql_literal(model_version())}
              AND description_hash IN ({','.join(map(sql_literal, part))})
        """)
        for key, vector in rows:
            found[key] = json.loads(vector)
    logger.info(
        "EMBEDDING CACHE LOOKUP: unique_descriptions=%d hits=%d misses=%d",
        len(hashes),
        len(found),
        len(hashes) - len(found),
    )
    return found


def save_embeddings(embedding_map):
    """Upsert embeddings into Delta in batches instead of one row at a time."""
    import logging
    import time

    logger = logging.getLogger(__name__)
    batch_size = 25
    now = datetime.now(timezone.utc).isoformat()

    entries = list(embedding_map.items())
    total = len(entries)

    logger.info("EMBEDDING CACHE WRITE START: total=%d", total)
    start_time = time.perf_counter()

    for offset in range(0, total, batch_size):
        batch = entries[offset:offset + batch_size]
        rows = []

        for key, vector in batch:
            if len(vector) != 384:
                raise ValueError("Expected 384-dimensional MiniLM embeddings")

            values = ",".join(
                f"CAST({float(x):.9g} AS FLOAT)"
                for x in vector
            )

            rows.append(
                f"SELECT "
                f"{sql_literal(key)} AS description_hash, "
                f"{sql_literal(MODEL_NAME)} AS model_name, "
                f"{sql_literal(model_version())} AS model_version, "
                f"array({values}) AS embedding, "
                f"TIMESTAMP {sql_literal(now)} AS created_at_utc"
            )

        # Combine rows into one source dataset for the MERGE.
        source_sql = "\nUNION ALL\n".join(rows)

        run_sql(f"""
            MERGE INTO {SCHEMA}.cache_contract_embeddings AS t
            USING ({source_sql}) AS s
            ON t.description_hash = s.description_hash
               AND t.model_name = s.model_name
               AND t.model_version = s.model_version
            WHEN NOT MATCHED THEN INSERT *
        """)

        logger.info(
            "EMBEDDING CACHE WRITE PROGRESS: %d/%d",
            min(offset + batch_size, total),
            total,
        )

    logger.info(
        "EMBEDDING CACHE WRITE COMPLETE: total=%d duration=%.1fs",
        total,
        time.perf_counter() - start_time,
    )
