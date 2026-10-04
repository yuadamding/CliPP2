"""Bounded, transient candidate evidence; SQLite is scratch, never a fit result.

Rows decode to the legacy dictionaries. Exact parameter text, proposal keys,
and source identities are interned on disk. Digests never establish equality
for interning: each UNIQUE constraint compares the complete stored value.
"""

import json
from pathlib import Path
import sqlite3
import tempfile


class CandidateStoreError(RuntimeError):
    """Scratch evidence could not be read or written; publication must stop."""


_FLAGS = ("selected_for_k", "selected_for_replicate_k", "selected", "publication_eligible")
_SOURCE = ("parent_requested_k", "parent_replicate", "parent_partition_sha256")
_INDEXED = (
    "replicate",
    "candidate_id",
    "requested_k",
    "status",
    "candidate_kind",
    "partition_sha256",
    "bic",
    "num_clusters",
    *_FLAGS,
)
_SELECT = """SELECT c.payload, c.replicate, c.candidate_id,
 c.selected_for_k, c.selected_for_replicate_k, c.selected, c.publication_eligible,
 p.parameters, s.parent_requested_k, s.parent_replicate, s.parent_partition_sha256,
 t.chain_cuts, t.proposal_partition_sha256
 FROM candidates c
 LEFT JOIN parameters p ON c.parameters_id = p.id
 LEFT JOIN sources s ON c.source_id = s.id
 LEFT JOIN proposals t ON c.proposal_id = t.id"""


def _json_default(value):
    # NumPy scalar inputs occur before DataFrame construction in the old path.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported candidate value: {type(value).__name__}")


def _json(value):
    # Python's float representation round-trips float64, including signed zero.
    # Nonfinite scalars are retained so the independent verifier can reject them.
    return json.dumps(value, separators=(",", ":"), default=_json_default)


class CandidateStore:
    """One owned SQLite file with bounded cache, transactions and ordered cursors.

    An explicit path must not exist. Without one, a unique file is created in
    directory (or the system temporary directory). close() removes owned scratch
    unless remove_on_close=False. Rows and lookup results are detached dictionaries;
    edits must be persisted with update().
    """

    def __init__(
        self,
        path=None,
        *,
        directory=None,
        cache_bytes=8 * 1024**2,
        transaction_rows=256,
        remove_on_close=True,
    ):
        if cache_bytes < 1024 or transaction_rows < 1:
            raise ValueError("Positive SQLite cache and transaction limits required")
        if path is not None and directory is not None:
            raise ValueError("Choose a store path or directory, not both")
        if path is None:
            import os

            fd, name = tempfile.mkstemp(prefix="clipp-candidates-", suffix=".sqlite", dir=directory)
            os.close(fd)
            self.path = Path(name)
        else:
            self.path = Path(path)
            with self.path.open("xb"):
                pass
        self.remove_on_close = remove_on_close
        self.transaction_rows = int(transaction_rows)
        self._writes = 0
        self._next_ids = {}
        self._closed = False
        self._failed = False
        try:
            self._db = sqlite3.connect(self.path, cached_statements=64)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA mmap_size=0")
            self._db.execute(f"PRAGMA cache_size=-{max(1, int(cache_bytes) // 1024)}")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript("""
                CREATE TABLE parameters(id INTEGER PRIMARY KEY, parameters TEXT NOT NULL UNIQUE);
                CREATE TABLE sources(id INTEGER PRIMARY KEY, parent_requested_k INTEGER NOT NULL,
                    parent_replicate INTEGER NOT NULL, parent_partition_sha256 TEXT NOT NULL,
                    UNIQUE(parent_requested_k, parent_replicate, parent_partition_sha256));
                CREATE TABLE proposals(id INTEGER PRIMARY KEY, chain_cuts TEXT NOT NULL,
                    proposal_partition_sha256 TEXT NOT NULL,
                    UNIQUE(chain_cuts, proposal_partition_sha256));
                CREATE INDEX proposal_digest ON proposals(proposal_partition_sha256,id);
                CREATE TABLE candidates(seq INTEGER PRIMARY KEY, replicate INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL, requested_k INTEGER, status TEXT,
                    candidate_kind TEXT, partition_sha256 TEXT, bic REAL, num_clusters INTEGER,
                    selected_for_k INTEGER, selected_for_replicate_k INTEGER,
                    selected INTEGER, publication_eligible INTEGER,
                    parameters_id INTEGER REFERENCES parameters(id),
                    source_id INTEGER REFERENCES sources(id), proposal_id INTEGER REFERENCES proposals(id),
                    payload TEXT NOT NULL, UNIQUE(replicate, candidate_id));
                CREATE INDEX candidate_budget ON candidates(replicate, requested_k, candidate_id);
                CREATE INDEX candidate_proposal ON candidates(proposal_id,replicate,requested_k,candidate_id);
                CREATE INDEX candidate_ranking ON candidates
                    (replicate, requested_k, status, publication_eligible, bic, num_clusters, candidate_id);
                CREATE TABLE visited(scope TEXT NOT NULL, key TEXT NOT NULL,
                    PRIMARY KEY(scope, key)) WITHOUT ROWID;
            """)
            self._db.commit()
        except (OSError, sqlite3.Error) as error:
            if hasattr(self, "_db"):
                self._db.close()
            self._closed = True
            raise CandidateStoreError(f"Cannot initialize candidate scratch {self.path}: {error}") from error

    def _execute(self, sql, values=()):
        if self._closed:
            raise CandidateStoreError("Candidate scratch is closed")
        if self._failed:
            raise CandidateStoreError("Candidate scratch previously failed; this attempt cannot continue")
        values = tuple(value.item() if hasattr(value, "item") else value for value in values)
        try:
            return self._db.execute(sql, values)
        except sqlite3.Error as error:
            self._failed = True
            raise CandidateStoreError(
                f"Candidate scratch operation failed at {self.path}: {error}"
            ) from error

    def _scratch_paths(self):
        return [self.path, *(Path(str(self.path) + suffix) for suffix in ("-journal", "-wal", "-shm"))]

    def flush(self):
        if self._closed:
            raise CandidateStoreError("Candidate scratch is closed")
        try:
            self._db.commit()
        except sqlite3.Error as error:
            self._failed = True
            raise CandidateStoreError(f"Cannot commit candidate scratch {self.path}: {error}") from error
        self._writes = 0

    def _changed(self):
        self._writes += 1
        if self._writes >= self.transaction_rows:
            self.flush()

    def _intern(self, table, columns, values):
        names = ",".join(columns)
        self._execute(
            f"INSERT OR IGNORE INTO {table} ({names}) VALUES ({','.join('?' for _ in values)})", values
        )
        condition = " AND ".join(name + "=?" for name in columns)
        return self._execute(f"SELECT id FROM {table} WHERE {condition}", values).fetchone()[0]

    def _values(self, record):
        payload = dict(record)
        parameter_id = source_id = proposal_id = None
        if "partition_parameters" in payload:
            parameters = payload.pop("partition_parameters")
            if not isinstance(parameters, str):
                raise TypeError("partition_parameters must be the exact serialized parameter string")
            parameter_id = self._intern("parameters", ("parameters",), (parameters,))
        if all(name in payload for name in _SOURCE):
            source_id = self._intern("sources", _SOURCE, tuple(payload.pop(name) for name in _SOURCE))
        proposal_fields = ("chain_cuts", "proposal_partition_sha256")
        if all(name in payload for name in proposal_fields):
            proposal_id = self._intern(
                "proposals", proposal_fields, tuple(payload.pop(name) for name in proposal_fields)
            )
        indexed = []
        for name in _INDEXED:
            value = record.get(name)
            if hasattr(value, "item"):
                value = value.item()
            if name in _FLAGS and value is not None:
                if not isinstance(value, bool):
                    raise TypeError(f"Candidate flag {name} must be Boolean")
                value = int(value)
            indexed.append(value)
        for name in ("replicate", "candidate_id", *_FLAGS):
            payload.pop(name, None)
        return (*indexed, parameter_id, source_id, proposal_id, _json(payload))

    def append(self, record, *, replicate=None):
        row = dict(record)
        for flag in _FLAGS[:3]:
            row.setdefault(flag, False)
        rep = row.get("replicate", 1) if replicate is None else replicate
        if "replicate" in row and row["replicate"] != rep:
            raise ValueError("Conflicting candidate replicate")
        if isinstance(rep, bool) or int(rep) != rep or rep < 1:
            raise ValueError("Candidate replicate must be a positive integer")
        rep = int(rep)
        next_id = self._next_ids.get(rep, 0)
        candidate_id = row.get("candidate_id", next_id)
        if isinstance(candidate_id, bool) or int(candidate_id) != candidate_id or candidate_id != next_id:
            raise ValueError(f"Expected candidate_id {next_id} for replicate {rep}")
        row.update(replicate=rep, candidate_id=next_id)
        names = (*_INDEXED, "parameters_id", "source_id", "proposal_id", "payload")
        values = self._values(row)
        self._execute(
            f"INSERT INTO candidates ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", values
        )
        self._next_ids[rep] = next_id + 1
        self._changed()
        return row

    def next_id(self, replicate=1):
        """Next deterministic per-replicate ID; reading does not reserve it."""
        if isinstance(replicate, bool) or int(replicate) != replicate or replicate < 1:
            raise ValueError("Candidate replicate must be a positive integer")
        return self._next_ids.get(int(replicate), 0)

    @staticmethod
    def _decode(row):
        result = json.loads(row["payload"])
        result.update(replicate=row["replicate"], candidate_id=row["candidate_id"])
        for name in _FLAGS:
            if row[name] is not None:
                result[name] = bool(row[name])
        if row["parameters"] is not None:
            result["partition_parameters"] = row["parameters"]
        if row["parent_requested_k"] is not None:
            result.update({name: row[name] for name in _SOURCE})
        if row["chain_cuts"] is not None:
            result.update(
                chain_cuts=row["chain_cuts"], proposal_partition_sha256=row["proposal_partition_sha256"]
            )
        return result

    def get(self, replicate, candidate_id):
        row = self._execute(
            _SELECT + " WHERE c.replicate=? AND c.candidate_id=?", (replicate, candidate_id)
        ).fetchone()
        if row is None:
            raise KeyError((replicate, candidate_id))
        return self._decode(row)

    def update(self, replicate, candidate_id, changes):
        if any(name in changes for name in ("replicate", "candidate_id")):
            raise ValueError("Candidate identity cannot be updated")
        row = self.get(replicate, candidate_id)
        row.update(changes)
        names = (*_INDEXED, "parameters_id", "source_id", "proposal_id", "payload")
        self._execute(
            "UPDATE candidates SET "
            + ",".join(name + "=?" for name in names)
            + " WHERE replicate=? AND candidate_id=?",
            (*self._values(row), replicate, candidate_id),
        )
        self._changed()
        return row

    @staticmethod
    def _where(filters):
        expressions, values = [], []
        for name, value in filters.items():
            if name not in (*_INDEXED, "proposal_partition_sha256"):
                raise ValueError(f"Unsupported candidate filter: {name}")
            if value is not None:
                if name == "proposal_partition_sha256":
                    expressions.append(
                        "c.proposal_id IN (SELECT id FROM proposals WHERE proposal_partition_sha256=?)"
                    )
                else:
                    expressions.append("c." + name + "=?")
                values.append(value)
        return expressions, values

    def iter_rows(self, **filters):
        expressions, values = self._where(filters)
        # A cursor observes a bounded prefix, even if its consumer appends rows.
        expressions.append("c.seq<=?")
        values.append(self._execute("SELECT COALESCE(MAX(seq),0) FROM candidates").fetchone()[0])
        cursor = self._execute(_SELECT + " WHERE " + " AND ".join(expressions) + " ORDER BY c.seq", values)
        try:
            for row in cursor:
                yield self._decode(row)
        except sqlite3.Error as error:
            self._failed = True
            raise CandidateStoreError(f"Cannot read candidate scratch {self.path}: {error}") from error
        finally:
            cursor.close()

    def count(self, **filters):
        expressions, values = self._where(filters)
        where = " WHERE " + " AND ".join(expressions) if expressions else ""
        return self._execute("SELECT COUNT(*) FROM candidates c" + where, values).fetchone()[0]

    def distinct_proposals(self, replicate=None):
        where = " AND c.replicate=?" if replicate is not None else ""
        values = (replicate,) if replicate is not None else ()
        return self._execute(
            "SELECT COUNT(*) FROM (SELECT t.proposal_partition_sha256 "
            "FROM proposals t INDEXED BY proposal_digest WHERE EXISTS "
            "(SELECT 1 FROM candidates c INDEXED BY candidate_proposal WHERE c.proposal_id=t.id"
            + where
            + ") GROUP BY t.proposal_partition_sha256)",
            values,
        ).fetchone()[0]

    def find_first(self, **filters):
        expressions, values = self._where(filters)
        where = " WHERE " + " AND ".join(expressions) if expressions else ""
        query = _SELECT
        if filters.get("proposal_partition_sha256") is not None:
            query = query.replace("FROM candidates c", "FROM candidates c INDEXED BY candidate_proposal")
        row = self._execute(query + where + " ORDER BY c.seq LIMIT 1", values).fetchone()
        return None if row is None else self._decode(row)

    def best(self, replicate, requested_k, *, eligible=True):
        where = " WHERE c.replicate=? AND c.requested_k=? AND c.status='scored'"
        if eligible:
            where += " AND c.publication_eligible=1"
        row = self._execute(
            _SELECT + where + " ORDER BY c.bic,c.num_clusters,c.candidate_id LIMIT 1",
            (replicate, requested_k),
        ).fetchone()
        return None if row is None else self._decode(row)

    def visited_add(self, scope, key):
        cursor = self._execute(
            "INSERT OR IGNORE INTO visited(scope,key) VALUES (?,?)", (_json(scope), _json(key))
        )
        self._changed()
        return cursor.rowcount == 1

    def visited_contains(self, scope, key):
        return (
            self._execute(
                "SELECT 1 FROM visited WHERE scope=? AND key=?", (_json(scope), _json(key))
            ).fetchone()
            is not None
        )

    def visited_discard(self, scope, key):
        self._execute("DELETE FROM visited WHERE scope=? AND key=?", (_json(scope), _json(key)))
        self._changed()

    def close(self, *, remove=None):
        if self._closed:
            return
        try:
            if self._failed:
                # A failed transaction may already have rolled back. Do not mask
                # the original disk-full/I/O exception with another commit.
                try:
                    self._db.rollback()
                except sqlite3.Error:
                    pass
            else:
                self.flush()
        finally:
            self._db.close()
            self._closed = True
        if self.remove_on_close if remove is None else remove:
            for path in self._scratch_paths():
                path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
