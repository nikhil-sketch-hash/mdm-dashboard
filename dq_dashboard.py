# dq_dashboard.py
import streamlit as st
import pandas as pd
import yaml
import os
import json
import datetime
import sqlite3
import re
import glob
from copy import deepcopy
from rapidfuzz import fuzz, process
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import (
    is_integer_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
)
import plotly.express as px
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Config files & defaults -------------------------
RULES_FILE = "rules.yaml"
SYN_FILE = "synonyms.yaml"
METADB = "dq_metadata.db"
PDO_FILE = "pdo.yaml"

DEFAULT_RULES = {
    "customer": {
        "required_columns": ["id", "name", "email", "phone"],
        "rules_per_column": {
            "id": ["not_null"],
            "name": ["not_null"],
            "email": ["email_format_check"],
            "phone": ["phone_format_check"],
        },
        "regexes": {
            "email": r"[^@]+@[^@]+\.[^@]+",
            "phone": r"^\+?\d{7,15}$",
        },
    },
    "finance": {
        "required_columns": ["id", "account_number", "balance", "date"],
        "rules_per_column": {
            "id": ["not_null"],
            "account_number": ["not_null"],
            "balance": ["not_null"],
            "date": ["not_null"],
        },
        "regexes": {},
    },
    "healthcare": {
        "required_columns": ["id", "patient_name", "diagnosis", "treatment"],
        "rules_per_column": {
            "id": ["not_null"],
            "patient_name": ["not_null"],
        },
        "regexes": {},
    },
    "product": {
        "required_columns": ["product_id", "product_name", "price"],
        "rules_per_column": {
            "product_id": ["not_null"],
            "product_name": ["not_null"],
            "price": ["not_null"],
        },
        "regexes": {},
    },
}

DEFAULT_SYNS = {
    "id": [
        "id",
        "customer_id",
        "account_id",
        "transaction_id",
        "patient_id",
        "product_id",
    ],
    "name": [
        "name",
        "full_name",
        "first_name",
        "last_name",
        "customer_name",
    ],
    "email": ["email", "emailid", "email_id", "mail", "mailid"],
    "phone": [
        "phone",
        "phone1",
        "phone_1",
        "phone2",
        "phone_2",
        "mobile",
        "mobile_number",
    ],
    "address": ["address", "addr", "street", "street_address", "location"],
    "city": ["city", "town", "municipality"],
    "state": ["state", "province", "region"],
    "zipcode": ["zip", "zipcode", "postal", "postal_code", "pincode"],
}

DEFAULT_PDO = {"connections": {}, "pdos": {}}

# ensure config files existence
def ensure_defaults():
    if not os.path.exists(RULES_FILE):
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_RULES, f)
    if not os.path.exists(SYN_FILE):
        with open(SYN_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_SYNS, f)
    if not os.path.exists(PDO_FILE):
        with open(PDO_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_PDO, f)

ensure_defaults()

# load/save yaml helpers
def load_yaml(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or deepcopy(fallback)
    except Exception:
        return deepcopy(fallback)

def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

rules_master = load_yaml(RULES_FILE, DEFAULT_RULES)
synonyms_master = load_yaml(SYN_FILE, DEFAULT_SYNS)
pdo_master = load_yaml(PDO_FILE, DEFAULT_PDO)

# --- Cleanup unnamed or invalid connections on startup ---
try:
    conns = pdo_master.get("connections", {})
    if conns:
        invalid_keys = [k for k in conns.keys() if not k or str(k).strip() == ""]
        for k in invalid_keys:
            del conns[k]
        if invalid_keys:
            pdo_master["connections"] = conns
            save_yaml(PDO_FILE, pdo_master)
            logger.info(f"Deleted {len(invalid_keys)} unnamed connection(s) from PDO on startup.")
except Exception as e:
    logger.warning(f"Error cleaning invalid connections: {e}")

# ------------------------- Metadata DB (with migration) -------------------------
def init_metadb():
    conn = sqlite3.connect(METADB, check_same_thread=False)
    cur = conn.cursor()
    # create trends table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            file TEXT,
            column_name TEXT,
            invalids INTEGER,
            rows INTEGER,
            valid_pct REAL
        )
    """
    )
    conn.commit()
    # add ts column if missing
    cur.execute("PRAGMA table_info(trends)")
    cols = [r[1] for r in cur.fetchall()]
    if "ts" not in cols:
        try:
            cur.execute("ALTER TABLE trends ADD COLUMN ts TEXT")
            conn.commit()
        except Exception:
            pass

    # merges legacy table (kept for compatibility)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS merges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            domain TEXT,
            file TEXT,
            method TEXT,
            columns TEXT,
            rows_merged INTEGER,
            output_file TEXT
        )
    """
    )
    conn.commit()

    # merge_history and cross_reference
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merge_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merge_group_id TEXT,
            domain TEXT,
            file TEXT,
            method TEXT,
            rows_merged INTEGER,
            output_file TEXT,
            ts TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cross_reference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merge_id INTEGER,
            domain TEXT,
            file TEXT,
            survivor_rowid TEXT,
            merged_rowid TEXT,
            ts TEXT
        )
    """)
    conn.commit()

    # Entities with version control
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT UNIQUE,
            domain TEXT,
            source_type TEXT,
            source_identifier TEXT,
            current_version INTEGER DEFAULT 1,
            record_count INTEGER,
            last_updated TEXT,
            schema_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            version INTEGER,
            record_count INTEGER,
            source_identifier TEXT,
            created_at TEXT,
            schema_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_data (
            entity_name TEXT,
            version INTEGER,
            data_json TEXT
        )
    """)
    conn.commit()
    return conn

DB_CONN = init_metadb()

# ------------------------- Domain classifier -------------------------
def train_domain_classifier():
    training = [
        ("account,balance,transaction,date", "finance"),
        ("patient,diagnosis,treatment,hospital", "healthcare"),
        ("customer,email,phone,address,city,state,zipcode,name", "customer"),
        ("product,product_id,price,category", "product"),
    ]
    df = pd.DataFrame(training, columns=["columns", "domain"])
    model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
    model.fit(df["columns"], df["domain"])
    return model

DOMAIN_MODEL = train_domain_classifier()

# ------------------------- Helpers -------------------------
def sanitize_dataframe(df):
    df = df.copy()
    for col in df.columns:
        try:
            if is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif is_bool_dtype(df[col]):
                df[col] = df[col].astype("bool").astype("int")
            elif is_string_dtype(df[col]) or is_object_dtype(df[col]):
                df[col] = df[col].astype(str).replace("nan", "").replace("None", "")
        except Exception:
            df[col] = df[col].astype(str)
    return df

def extract_best(res):
    # process.extractOne may return (best, score, idx) or (best, score)
    if not res:
        return None, 0
    try:
        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) >= 2:
                return res[0], float(res[1])
    except Exception:
        pass
    return None, 0

def find_best_match(required_col, candidates, synonyms=None, threshold=80):
    candidates = list(candidates)
    variants = [required_col] if required_col else []
    if synonyms:
        for key, vals in synonyms.items():
            try:
                if required_col.lower() == key.lower() or required_col.lower() in [
                    v.lower() for v in vals
                ]:
                    variants.extend(vals)
            except Exception:
                pass
    variants = [v for v in set(variants) if isinstance(v, str)]
    for v in variants:
        try:
            res = process.extractOne(v, candidates, scorer=fuzz.ratio)
        except TypeError:
            # fallback if signatures differ
            try:
                res = process.extractOne(v, candidates)
            except Exception:
                res = None
        name, sc = extract_best(res)
        if name and sc >= threshold:
            return name
    return None

def detect_domain(df_columns, rules, synonyms):
    # Count how many required columns match for each domain
    scores = {}
    for domain, defn in rules.items():
        cnt = 0
        for req in defn.get("required_columns", []):
            for col in df_columns:
                if find_best_match(req, [col], synonyms):
                    cnt += 1
                    break
        scores[domain] = cnt
    if not scores:
        return "unknown"
    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else "unknown"

def natural_language_to_expression(text):
    text = text.lower().strip()
    
    # Not null checks
    if "not null" in text or "cannot be null" in text or "is not empty" in text:
        return "value is None or len(str(value).strip()) == 0"
    
    # Length checks
    match = re.search(r"length (is|should be) (greater than|at least) (\d+)", text)
    if match:
        return f"len(str(value).strip()) <= {match.group(3)}"
    
    match = re.search(r"length (is|should be) (less than|at most) (\d+)", text)
    if match:
        return f"len(str(value).strip()) >= {match.group(3)}"
    match = re.search(r"length is exactly (\d+)", text)
    if match:
        return f"len(str(value).strip()) != {match.group(1)}"
        
    match = re.search(r"length is between (\d+) and (\d+)", text)
    if match:
        min_len, max_len = int(match.group(1)), int(match.group(2))
        return f"len(str(value).strip()) < {min_len} or len(str(value).strip()) > {max_len}"
    # Value checks
    if "value is numeric" in text or "must be numeric" in text:
        return "not str(value).isdigit()"
    if "value is a date" in text or "must be a date" in text:
        return "pd.to_datetime(value, errors='coerce').isnull()"
    
    return None

# ------------------------- Rules engine -------------------------
# supported patterns:
# - "not_null"
# - "email_format_check" (uses domain regexes or default)
# - "phone_format_check"
# - "numeric_range:min:max"
# - "date_format:fmt"
# - "regex:pattern"
# - "ifelse:python_expr" -> if expression evaluates True => mark INVALID
def apply_rules(
    domain, df, rules, synonyms, selected_actual_columns=None, fuzzy_threshold=85, sample_limit=20000
):
    issues = {}
    matched_map = {}
    invalid_counts = {}
    domain_def = rules.get(domain, {})
    rules_map = domain_def.get("rules_per_column", {})
    regexes = domain_def.get("regexes", {})
    # map canonical -> actual columns
    for canonical, rlist in rules_map.items():
        actual = (
            canonical
            if canonical in df.columns
            else find_best_match(canonical, df.columns.tolist(), synonyms)
        )
        if actual:
            matched_map.setdefault(actual, {"canonical": canonical, "rules": []})
            if isinstance(rlist, list):
                matched_map[actual]["rules"].extend(rlist)
            else:
                matched_map[actual]["rules"].append(rlist)
    # if user selected a subset of actual columns to check, filter matched_map
    if selected_actual_columns:
        matched_map = {a: meta for a, meta in matched_map.items() if a in selected_actual_columns}
    # evaluate on sample for performance
    df_eval = df if len(df) <= sample_limit else df.sample(n=sample_limit, random_state=42)
    for actual, meta in matched_map.items():
        rules_list = meta.get("rules", [])
        canonical = meta.get("canonical")
        invalid_idx = set()
        for rule in rules_list:
            try:
                if rule == "not_null":
                    mask = df_eval[actual].isnull() | (df_eval[actual].astype(str).str.strip() == "")
                    invalid_idx.update(df_eval[mask].index.tolist())
                elif rule == "email_format_check":
                    pat = regexes.get(canonical) or regexes.get("email") or r"[^@]+@[^@]+\.[^@]+"
                    mask = ~df_eval[actual].fillna("").astype(str).str.match(pat, na=False)
                    invalid_idx.update(df_eval[mask].index.tolist())
                elif rule == "phone_format_check":
                    pat = regexes.get(canonical) or regexes.get("phone") or r"^\+?\d{7,15}$"
                    mask = ~df_eval[actual].fillna("").astype(str).str.match(pat, na=False)
                    invalid_idx.update(df_eval[mask].index.tolist())
                elif isinstance(rule, str) and rule.startswith("numeric_range:"):
                    try:
                        _, mn, mx = rule.split(":")
                        numeric = pd.to_numeric(df_eval[actual], errors="coerce")
                        mask = numeric.isnull() | (numeric < float(mn)) | (numeric > float(mx))
                        invalid_idx.update(df_eval[mask].index.tolist())
                    except Exception:
                        pass
                elif isinstance(rule, str) and rule.startswith("date_format:"):
                    fmt = rule.split(":", 1)[1]
                    parsed = pd.to_datetime(df_eval[actual], format=fmt, errors="coerce")
                    mask = parsed.isnull()
                    invalid_idx.update(df_eval[mask].index.tolist())
                elif isinstance(rule, str) and rule.startswith("regex:"):
                    pat = rule.split(":", 1)[1]
                    mask = ~df_eval[actual].fillna("").astype(str).str.match(pat, na=False)
                    invalid_idx.update(df_eval[mask].index.tolist())
                elif isinstance(rule, str) and rule.startswith("ifelse:"):
                    expr = rule.split(":", 1)[1]
                    # For each value evaluate expression in restricted env; if True => mark INVALID
                    for idx, v in zip(df_eval.index.tolist(), df_eval[actual].tolist()):
                        try:
                            # IMPORTANT: Pass required built-ins to enable user expressions
                            safe_globals = {
                                "__builtins__": {
                                    "len": len,
                                    "str": str,
                                    "int": int,
                                    "float": float,
                                    "bool": bool,
                                    "isinstance": isinstance,
                                    "pd": pd,
                                },
                            }
                            safe_locals = {"value": v, "re": re}
                            res = eval(expr, safe_globals, safe_locals)
                            if bool(res):
                                invalid_idx.add(idx)
                        except Exception as e:
                            # Catches errors like NameError (if a function isn't passed) and marks as invalid
                            logger.error(f"Expression evaluation error: {e}")
                            invalid_idx.add(idx)
                else:
                    # unknown rule token - skip
                    pass
            except Exception as e:
                logger.warning(f"Error applying rule {rule} on {actual}: {e}")
        invalid_counts[actual] = len(invalid_idx)
        if invalid_idx:
            issues[f"Invalid - {actual} ({len(invalid_idx)})"] = df.loc[sorted(invalid_idx)]
    return issues, matched_map, invalid_counts

# ------------------------- Duplicate detection and merge -------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""
    # remove punctuation, lower, trim
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)

def detect_potential_duplicates(df, match_cols, threshold=85):
    """
    Informatica-style potential duplicate detection.
    - Builds a composite, token-sorted key combining selected columns.
    - Uses blocking for scalability.
    - Performs weighted fuzzy scoring per attribute and on the composite key.
    - Returns list of pairs (row_idx_a, row_idx_b) and subset dataframe of matched rows.
    """
    pairs = []
    seen = set()

    compare_cols = list(match_cols) if match_cols else list(df.columns)
    if not compare_cols:
        return [], pd.DataFrame()

    # Build composite normalized key per row
    def build_composite_key_series(df_local, cols):
        def _build(row):
            vals = []
            for c in cols:
                try:
                    v = normalize_text(row[c])
                    if v:
                        vals.append(v)
                except Exception:
                    continue
            # token-sort style: split tokens and sort set
            tokens = sorted(set(" ".join(vals).split()))
            return " ".join(tokens)
        return df_local.apply(_build, axis=1)

    combined_series = build_composite_key_series(df, compare_cols)

    # Blocking key - use first 3 chars of composite key (or empty)
    try:
        blocking_key = combined_series.str[:3]
    except Exception:
        blocking_key = pd.Series([""] * len(df))

    for block_val in blocking_key.unique():
        block_idxs = blocking_key[blocking_key == block_val].index.tolist()
        for i in range(len(block_idxs)):
            for j in range(i + 1, len(block_idxs)):
                idx_i, idx_j = block_idxs[i], block_idxs[j]
                # Weighted scoring
                total_score = 0.0
                weight_sum = 0.0
                # Per-column scores
                for col in compare_cols:
                    try:
                        val_i = normalize_text(df.at[idx_i, col])
                        val_j = normalize_text(df.at[idx_j, col])
                        if val_i and val_j:
                            sc = fuzz.token_sort_ratio(val_i, val_j)
                            # weight certain columns higher
                            if re.search(r"name|email|phone|id", re.IGNORECASE):
                                w = 1.5
                            elif re.search(r"email|phone", col, re.IGNORECASE):
                                w = 1.5
                            else:
                                w = 1.0
                            total_score += sc * w
                            weight_sum += w
                    except Exception:
                        continue

                # Also include composite key match (helps with swapped fields)
                try:
                    comp_sc = fuzz.token_sort_ratio(combined_series.iloc[idx_i], combined_series.iloc[idx_j])
                    total_score += comp_sc * 1.2
                    weight_sum += 1.2
                except Exception:
                    pass

                if weight_sum == 0:
                    continue

                avg_score = total_score / weight_sum

                if avg_score >= threshold:
                    pairs.append((idx_i, idx_j))
                    seen.update([idx_i, idx_j])

    return pairs, df.loc[list(seen)] if seen else pd.DataFrame()

def merge_records_auto(group):
    result = {}
    for col in group.columns:
        vals = group[col].dropna().tolist()
        if pd.api.types.is_numeric_dtype(group[col]):
            try:
                result[col] = group[col].mean()
            except Exception:
                result[col] = vals[0] if vals else None
        elif len(vals) == 1:
            result[col] = vals[0]
        else:
            try:
                mode = group[col].mode()
                result[col] = mode.iloc[0] if not mode.empty else (vals[0] if vals else None)
            except Exception:
                result[col] = vals[0] if vals else None
    return pd.DataFrame([result])

def fuzzy_groups_for_column(df, col, threshold=85):
    vals = df[col].dropna().astype(str).unique().tolist()
    checked = set()
    groups = []
    for v in vals:
        if v in checked:
            continue
        matches = process.extract(v, vals, scorer=fuzz.token_sort_ratio, limit=None)
        cluster = [m[0] for m in matches if m[1] >= threshold]
        if len(cluster) > 1:
            groups.append(cluster)
            checked.update(cluster)
    return groups

# ------------------------- Connections & Table browsing -------------------------
def test_sqlalchemy_url(url):
    try:
        engine = create_engine(url)
        conn = engine.connect()
        conn.close()
        return True, engine
    except Exception as e:
        return False, str(e)

def list_tables_for_conn(url):
    try:
        engine = create_engine(url)
        insp = inspect(engine)
        tables = insp.get_table_names()
        views = insp.get_view_names()
        return tables + views
    except Exception as e:
        logger.warning(f"list_tables_for_conn error: {e}")
        return []

# ------------------------- Small UI helpers -------------------------
def dq_color_pct(val):
    try:
        v = float(val)
    except Exception:
        return "black"
    if v >= 80:
        return "green"
    if v >= 50:
        return "orange"
    return "red"

# ------------------------- Streamlit app -------------------------
st.set_page_config("MDM & Data Quality Dashboard", layout="wide")
st.title("MDM & Data Quality Dashboard")
# Initialize session state for persistent data
if "processed" not in st.session_state:
    st.session_state.processed = []
if "active_df" not in st.session_state:
    st.session_state.active_df = pd.DataFrame()

# -------- Connections tab --------
tabs = st.tabs(
    [
        "Connections",
        "Tables/PDO",
        "Entities",
        "Upload & Process",
        "Profiling",
        "Rules Manager",
        "Deduplication",
        "Merge History",
        "Trends",
    ]
)

with tabs[0]:
    st.header("Connections Manager")
    st.markdown("Add, test, update, or delete your saved connections.")

    conn_types = [
        "SQLite",
        "Postgres",
        "MySQL",
        "SQL Server",
        "Oracle",
        "Snowflake",
        "FlatFile",
        "Other (SQLAlchemy)",
    ]

    # --- ADD NEW CONNECTION ---
    st.subheader("➕ Add New Connection")
    new_ct = st.selectbox("DB Type", conn_types, index=0, key="conn_type_add")

    col1, col2 = st.columns(2)
    with col1:
        conn_name = st.text_input("Connection Name", key="conn_name_add")
    with col2:
        conn_str_input = st.text_input(
            "SQLAlchemy URL (for DB) or Folder Path (for FlatFile)",
            placeholder="e.g. sqlite:///my.db or /data/files/",
            key="conn_str_add",
        )

    if st.button("Test & Save Connection"):
        if new_ct.lower() in ["flatfile", "flat file", "file", "csv"]:
            # Flat file connection validation
            if os.path.isdir(conn_str_input):
                pdo_master.setdefault("connections", {})[conn_name] = {
                    "type": new_ct,
                    "path": conn_str_input,
                }
                save_yaml(PDO_FILE, pdo_master)
                st.success(f"Flat file connection '{conn_name}' saved successfully.")
            else:
                st.error("Invalid folder path for flat file connection.")
        else:
            # Database connection
            ok, eng = test_sqlalchemy_url(conn_str_input)
            if ok:
                pdo_master.setdefault("connections", {})[conn_name] = {
                    "type": new_ct,
                    "sqlalchemy_url": conn_str_input,
                }
                save_yaml(PDO_FILE, pdo_master)
                st.success(f"Database connection '{conn_name}' saved successfully.")
            else:
                st.error(f"Connection failed: {eng}")

    st.markdown("---")

    # --- UPDATE / DELETE EXISTING CONNECTIONS ---
    st.subheader("✏️ Manage Existing Connections")

    conns = pdo_master.get("connections", {})
    if not conns:
        st.info("No saved connections. Add one above.")
    else:
        sel_conn = st.selectbox("Select a connection to view/update", [None] + list(conns.keys()))
        if sel_conn:
            meta = conns[sel_conn]
            conn_type = meta.get("type", "")
            st.write(f"**Connection Type:** {conn_type}")

            # FlatFile connections
            if any(t in conn_type.lower() for t in ["flatfile", "flat file", "file", "csv"]):
                current_path = meta.get("path", "")
                new_path = st.text_input(
                    "Update folder path",
                    value=current_path,
                    help="Enter a valid folder path for this flat file connection",
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Test Connection"):
                        if os.path.isdir(new_path):
                            st.success(f"✅ Folder is accessible: {new_path}")
                        else:
                            st.error("❌ Folder not found or inaccessible.")
                with col2:
                    if st.button("Update Connection"):
                        if not os.path.isdir(new_path):
                            st.error("Invalid folder path.")
                        else:
                            meta["path"] = new_path
                            pdo_master["connections"][sel_conn] = meta
                            save_yaml(PDO_FILE, pdo_master)
                            st.success(f"Connection '{sel_conn}' updated successfully.")
                            st.rerun()
                with col3:
                    if st.button("🗑️ Delete Connection"):
                        del pdo_master["connections"][sel_conn]
                        save_yaml(PDO_FILE, pdo_master)
                        st.warning(f"Connection '{sel_conn}' deleted.")
                        st.rerun()

            # Database connections
            else:
                current_url = meta.get("sqlalchemy_url", "")
                new_url = st.text_input(
                    "Update SQLAlchemy URL",
                    value=current_url,
                    help="Enter the new SQLAlchemy connection string.",
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Test Connection"):
                        ok, msg = test_sqlalchemy_url(new_url)
                        if ok:
                            st.success("✅ Connection test successful.")
                        else:
                            st.error(f"❌ Connection test failed: {msg}")
                with col2:
                    if st.button("Update Connection"):
                        ok, msg = test_sqlalchemy_url(new_url)
                        if ok:
                            meta["sqlalchemy_url"] = new_url
                            pdo_master["connections"][sel_conn] = meta
                            save_yaml(PDO_FILE, pdo_master)
                            st.success(f"Connection '{sel_conn}' updated successfully.")
                            st.rerun()
                        else:
                            st.error(f"Update failed: {msg}")
                with col3:
                    if st.button("🗑️ Delete Connection"):
                        del pdo_master["connections"][sel_conn]
                        save_yaml(PDO_FILE, pdo_master)
                        st.warning(f"Connection '{sel_conn}' deleted.")
                        st.rerun()

    st.markdown("---")

    # --- VIEW ALL CONNECTIONS ---
    st.subheader("📋 Saved Connections")
    st.json(pdo_master.get("connections", {}))

# -------- Tables/PDO tab --------
with tabs[1]:
    st.header("Tables / PDOs")
    conns = pdo_master.get("connections", {})
    if not conns:
        st.info("No saved connections. Add one in Connections tab.")
    else:
        sel = st.selectbox("Choose connection", [None] + list(conns.keys()))
        if sel:
            meta = conns[sel]
            st.write("Type:", meta.get("type"))
            conn_type = meta.get("type", "").lower()

            if any(t in conn_type for t in ["flatfile", "flat file", "csv", "file"]):
                # Auto-detect base directory from PDO connection
                base_path = meta.get("path") or meta.get("folder") or ""
                if not base_path or not os.path.isdir(base_path):
                    st.warning("⚠️ Base folder path not found or invalid in the connection settings.")
                    st.info("Please update your flat file connection in Connections tab with a valid 'path' key.")
                else:
                    st.success(f"Using base folder from connection: `{base_path}`")

                    pattern = st.text_input(
                        "File name pattern (use * wildcard)",
                        value="*.csv",
                        help="Example: customer_*.csv or *.parquet"
                    )

                    if st.button("List matching files"):
                        try:
                            files = glob.glob(os.path.join(base_path, pattern))
                            if not files:
                                st.warning("No files matched that pattern.")
                            else:
                                st.session_state["_flatfile_list"] = sorted(files)
                                st.success(f"Found {len(files)} files.")
                        except Exception as e:
                            st.error(f"Error scanning directory: {e}")

                    files = st.session_state.get("_flatfile_list", [])
                    chosen_file = st.selectbox("Choose file to preview", [None] + files)

                    if chosen_file:
                        limit = st.number_input("Preview rows", min_value=5, max_value=10000, value=100, step=5)
                        try:
                            if chosen_file.endswith(".csv"):
                                df_preview = pd.read_csv(chosen_file, nrows=limit)
                            elif chosen_file.endswith(".parquet"):
                                df_preview = pd.read_parquet(chosen_file)
                            else:
                                df_preview = pd.read_excel(chosen_file, nrows=limit)

                            st.dataframe(df_preview)

                            if st.button("Load into session for profiling"):
                                st.session_state["active_df"] = df_preview
                                st.session_state["active_source"] = {
                                    "type": "flatfile",
                                    "path": chosen_file,
                                    "connection": sel,
                                }
                                st.success(f"Loaded {os.path.basename(chosen_file)} into active_df")
                        except Exception as e:
                            st.error(f"Could not preview file: {e}")
            else:
                if st.button("List tables/views"):
                    names = list_tables_for_conn(meta.get("sqlalchemy_url"))
                    st.session_state["_last_table_list"] = names
                names = st.session_state.get("_last_table_list", [])
                pattern = st.text_input("Search pattern (use * wildcard)", value="*", key="table_search")
                regex = re.compile(pattern.replace("*", ".*"), re.IGNORECASE)
                filtered = [n for n in names if regex.search(n)]
                chosen = st.selectbox("Choose table/view", [None] + filtered)
                if chosen:
                    limit = st.number_input(
                        "Preview rows", min_value=5, max_value=10000, value=100, step=5
                    )
                    try:
                        df_preview = pd.read_sql(
                            text(f"SELECT * FROM {chosen} LIMIT {limit}"),
                            create_engine(meta.get("sqlalchemy_url")),
                        )
                        st.dataframe(df_preview)
                        if st.button("Load into session for profiling"):
                            st.session_state["active_df"] = df_preview
                            st.session_state["active_source"] = {
                                "type": "db",
                                "connection": sel,
                                "table": chosen,
                            }
                            st.success("Loaded table into active_df")
                    except Exception as e:
                        st.error(f"Could not preview table: {e}")

    st.markdown("Register PDO from file upload")
    uploaded = st.file_uploader("Upload file to register PDO", type=["csv", "parquet", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded, nrows=10)
            elif uploaded.name.endswith(".parquet"):
                df_up = pd.read_parquet(uploaded)
            else:
                df_up = pd.read_excel(uploaded, nrows=10)
            pdo_master.setdefault("pdos", {})[uploaded.name] = {
                "type": "file",
                "identifier": uploaded.name,
                "columns": [{"name": c} for c in df_up.columns],
            }
            save_yaml(PDO_FILE, pdo_master)
            st.success("Registered PDO from upload")
        except Exception as e:
            st.error(f"Error registering PDO: {e}")

# -------- Entities tab --------
with tabs[2]:
    st.header("MDM Entities — Persistent Datasets (with Version Control)")
    cur = DB_CONN.cursor()

    cur.execute("SELECT entity_name, domain, source_type, source_identifier, current_version, record_count, last_updated FROM entities ORDER BY last_updated DESC")
    rows = cur.fetchall()

    if not rows:
        st.info("No entities registered yet. Upload a file or process a table to create one.")
    else:
        df_entities = pd.DataFrame(rows, columns=["Entity", "Domain", "Source Type", "Source ID", "Version", "Rows", "Last Updated"])
        st.dataframe(df_entities, use_container_width=True)

        selected_entity = st.selectbox("Select an entity to view", [None] + df_entities["Entity"].tolist())
        if selected_entity:
            cur.execute("SELECT version, record_count, created_at FROM entity_versions WHERE entity_name = ? ORDER BY version DESC", (selected_entity,))
            versions = cur.fetchall()
            if not versions:
                st.warning("No versions found for this entity.")
            else:
                df_ver = pd.DataFrame(versions, columns=["Version", "Rows", "Created At"])
                st.subheader(f"Versions for Entity: {selected_entity}")
                st.dataframe(df_ver, use_container_width=True)

                sel_ver = st.selectbox("Select a version to preview", df_ver["Version"].tolist())
                if sel_ver:
                    cur.execute("SELECT data_json FROM entity_data WHERE entity_name = ? AND version = ?", (selected_entity, sel_ver))
                    rows = cur.fetchall()
                    data = [json.loads(r[0]) for r in rows]
                    df_entity = pd.DataFrame(data)

                    st.subheader(f"Entity Preview — {selected_entity}_v{sel_ver}")
                    st.dataframe(df_entity.head(200))

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Load this version into session for processing"):
                            st.session_state["active_df"] = df_entity
                            st.session_state["active_source"] = {"type": "entity", "name": selected_entity, "version": sel_ver}
                            st.success(f"Entity '{selected_entity}' v{sel_ver} loaded into session.")
                    with c2:
                        if st.button("🕓 Rollback to this version"):
                            cur.execute("UPDATE entities SET current_version = ? WHERE entity_name = ?", (sel_ver, selected_entity))
                            DB_CONN.commit()
                            st.success(f"Rolled back {selected_entity} to version {sel_ver}.")
                            st.rerun()

                # Compare two versions
                st.markdown("### Compare Versions")
                ver_options = df_ver["Version"].tolist()
                if len(ver_options) >= 2:
                    v1 = st.selectbox("Version A", ver_options, index=0, key="cmp_v1")
                    v2 = st.selectbox("Version B", ver_options, index=1, key="cmp_v2")
                    if st.button("Compare Versions"):
                        cur.execute("SELECT data_json FROM entity_data WHERE entity_name = ? AND version = ?", (selected_entity, v1))
                        rows1 = [json.loads(r[0]) for r in cur.fetchall()]
                        cur.execute("SELECT data_json FROM entity_data WHERE entity_name = ? AND version = ?", (selected_entity, v2))
                        rows2 = [json.loads(r[0]) for r in cur.fetchall()]
                        df1 = pd.DataFrame(rows1)
                        df2 = pd.DataFrame(rows2)
                        # Basic diffs: added, removed, changed
                        try:
                            # find common columns
                            common_cols = [c for c in df1.columns if c in df2.columns]
                            df1_id = df1.reset_index().rename(columns={"index": "__idx1"})
                            df2_id = df2.reset_index().rename(columns={"index": "__idx2"})
                            merged = df1.merge(df2, on=common_cols, how='outer', indicator=True)
                            added = merged[merged["_merge"] == "right_only"]
                            removed = merged[merged["_merge"] == "left_only"]
                            changed = pd.concat([df1[~df1.apply(tuple,1).isin(df2.apply(tuple,1))], df2[~df2.apply(tuple,1).isin(df1.apply(tuple,1))]])
                            st.subheader("Added rows in B (not in A)")
                            st.dataframe(added.head(200))
                            st.subheader("Removed rows in B (present in A but not in B)")
                            st.dataframe(removed.head(200))
                            st.subheader("Changed rows (heuristic)")
                            st.dataframe(changed.head(200))
                        except Exception as e:
                            st.error(f"Comparison failed: {e}")

# -------- Upload & Process tab --------
with tabs[3]:
    st.header("Upload CSV or use loaded table")
    uploaded_files = st.file_uploader(
        "Upload CSV(s)", accept_multiple_files=True, type=["csv"]
    )
    # New logic to handle uploaded files
    if uploaded_files:
        for f in uploaded_files:
            try:
                if f.name.endswith(".csv"):
                    df = pd.read_csv(f)
                elif f.name.endswith(".parquet"):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_excel(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue

            df = sanitize_dataframe(df)
            domain = detect_domain(df.columns.tolist(), rules_master, synonyms_master)
            if domain == "unknown":
                try:
                    domain = DOMAIN_MODEL.predict([",".join(df.columns).lower()])[0]
                except Exception:
                    domain = "unknown"
            issues, matched_map, invalids = apply_rules(
                domain, df, rules_master, synonyms_master
            )
            # profile summary
            prof = []
            for col in df.columns:
                prof.append(
                    {
                        "column": col,
                        "non_null": int(df[col].notnull().sum()),
                        "null_count": int(df[col].isnull().sum()),
                        "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                        "distinct": int(df[col].nunique()),
                        "dtype": str(df[col].dtype),
                        "applied_rules": ", ".join(
                            matched_map.get(col, {}).get("rules", [])
                        ),
                    }
                )
            # store in session_state.processed
            existing_entry_index = -1
            for i, entry in enumerate(st.session_state.processed):
                if entry["filename"] == f.name:
                    existing_entry_index = i
                    break
            new_entry = {
                "filename": f.name,
                "df": df,
                "domain": domain,
                "issues": issues,
                "matched_map": matched_map,
                "invalids": invalids,
                "profile": pd.DataFrame(prof),
            }
            if existing_entry_index != -1:
                st.session_state.processed[existing_entry_index] = new_entry
            else:
                st.session_state.processed.append(new_entry)

            # store trend per column in DB
            cur = DB_CONN.cursor()
            for col, inv in invalids.items():
                total = len(df)
                valid_pct = round(((total - inv) / total) * 100, 2) if total else 100.0
                cur.execute(
                    "INSERT INTO trends (ts, domain, file, column_name, invalids, rows, valid_pct) VALUES (?,?,?,?,?,?,?)",
                    (
                        datetime.datetime.now().isoformat(),
                        domain,
                        f.name,
                        col,
                        int(inv),
                        int(total),
                        float(valid_pct),
                    ),
                )
            DB_CONN.commit()
            st.success(f"Processed {f.name}")

            # --- Persist dataset as an MDM Entity with versioning ---
            try:
                entity_name = os.path.splitext(f.name)[0].lower()
                df_sample = df.copy()
                cur = DB_CONN.cursor()

                # Check if entity exists
                cur.execute("SELECT current_version FROM entities WHERE entity_name = ?", (entity_name,))
                row = cur.fetchone()
                if row:
                    new_version = row[0] + 1
                    cur.execute("UPDATE entities SET current_version = ? WHERE entity_name = ?", (new_version, entity_name))
                else:
                    new_version = 1

                schema_info = [{"name": c, "dtype": str(df_sample[c].dtype)} for c in df_sample.columns]
                now = datetime.datetime.now().isoformat()

                # Upsert metadata in entities
                cur.execute("""
                    INSERT OR REPLACE INTO entities (entity_name, domain, source_type, source_identifier, current_version, record_count, last_updated, schema_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_name,
                    domain,
                    "file",
                    f.name,
                    new_version,
                    len(df_sample),
                    now,
                    json.dumps(schema_info),
                ))

                # Record version in entity_versions table
                cur.execute("""
                    INSERT INTO entity_versions (entity_name, version, record_count, source_identifier, created_at, schema_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity_name,
                    new_version,
                    len(df_sample),
                    f.name,
                    now,
                    json.dumps(schema_info),
                ))

                # Store entity data
                data_records = df_sample.to_dict(orient="records")
                for row in data_records:
                    cur.execute(
                        "INSERT INTO entity_data (entity_name, version, data_json) VALUES (?, ?, ?)",
                        (entity_name, new_version, json.dumps(row))
                    )

                DB_CONN.commit()
                st.success(f"✅ Entity '{entity_name}' version {new_version} saved to MDM repository ({len(df_sample)} rows).")
            except Exception as e:
                st.error(f"Failed to persist versioned entity: {e}")

    # loaded active table from Tables tab
    if "active_df" in st.session_state and not st.session_state.active_df.empty:
        if st.checkbox("Process active loaded table"):
            df = st.session_state["active_df"]
            df = sanitize_dataframe(df)
            domain = detect_domain(df.columns.tolist(), rules_master, synonyms_master)
            if domain == "unknown":
                try:
                    domain = DOMAIN_MODEL.predict([",".join(df.columns).lower()])[0]
                except Exception:
                    domain = "unknown"
            issues, matched_map, invalids = apply_rules(
                domain, df, rules_master, synonyms_master
            )
            prof = []
            for col in df.columns:
                prof.append(
                    {
                        "column": col,
                        "non_null": int(df[col].notnull().sum()),
                        "null_count": int(df[col].isnull().sum()),
                        "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                        "distinct": int(df[col].nunique()),
                        "dtype": str(df[col].dtype),
                        "applied_rules": ", ".join(
                            matched_map.get(col, {}).get("rules", [])
                        ),
                    }
                )
            filename = st.session_state.get("active_source", {}).get("table", "active_table")
            
            existing_entry_index = -1
            for i, entry in enumerate(st.session_state.processed):
                if entry["filename"] == filename:
                    existing_entry_index = i
                    break
            new_entry = {
                "filename": filename,
                "df": df,
                "domain": domain,
                "issues": issues,
                "matched_map": matched_map,
                "invalids": invalids,
                "profile": pd.DataFrame(prof),
            }
            if existing_entry_index != -1:
                st.session_state.processed[existing_entry_index] = new_entry
            else:
                st.session_state.processed.append(new_entry)
            cur = DB_CONN.cursor()
            for col, inv in invalids.items():
                total = len(df)
                valid_pct = round(((total - inv) / total) * 100, 2) if total else 100.0
                cur.execute(
                    "INSERT INTO trends (ts, domain, file, column_name, invalids, rows, valid_pct) VALUES (?,?,?,?,?,?,?)",
                    (
                        datetime.datetime.now().isoformat(),
                        domain,
                        filename,
                        col,
                        int(inv),
                        int(total),
                        float(valid_pct),
                    ),
                )
            DB_CONN.commit()
            st.success("Processed active table")

            # Persist as entity (use filename or table name as identifier)
            try:
                source_id = st.session_state.get("active_source", {}).get("path") or filename
                entity_name = os.path.splitext(os.path.basename(source_id))[0].lower()
                df_sample = df.copy()
                cur = DB_CONN.cursor()
                cur.execute("SELECT current_version FROM entities WHERE entity_name = ?", (entity_name,))
                row = cur.fetchone()
                if row:
                    new_version = row[0] + 1
                    cur.execute("UPDATE entities SET current_version = ? WHERE entity_name = ?", (new_version, entity_name))
                else:
                    new_version = 1
                schema_info = [{"name": c, "dtype": str(df_sample[c].dtype)} for c in df_sample.columns]
                now = datetime.datetime.now().isoformat()
                cur.execute("""
                    INSERT OR REPLACE INTO entities (entity_name, domain, source_type, source_identifier, current_version, record_count, last_updated, schema_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_name,
                    domain,
                    st.session_state.get("active_source", {}).get("type", "db"),
                    source_id,
                    new_version,
                    len(df_sample),
                    now,
                    json.dumps(schema_info),
                ))
                cur.execute("""
                    INSERT INTO entity_versions (entity_name, version, record_count, source_identifier, created_at, schema_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity_name,
                    new_version,
                    len(df_sample),
                    source_id,
                    now,
                    json.dumps(schema_info),
                ))
                data_records = df_sample.to_dict(orient="records")
                for row in data_records:
                    cur.execute(
                        "INSERT INTO entity_data (entity_name, version, data_json) VALUES (?, ?, ?)",
                        (entity_name, new_version, json.dumps(row))
                    )
                DB_CONN.commit()
                st.success(f"✅ Entity '{entity_name}' version {new_version} saved to MDM repository ({len(df_sample)} rows).")
            except Exception as e:
                st.error(f"Failed to persist versioned entity: {e}")

# -------- Profiling tab --------
with tabs[4]:
    st.header("Profiling & Summary")
    if not st.session_state.processed:
        st.info("No datasets processed yet. Use Upload & Process tab.")
    else:
        # Generate a unique list of filenames for the selectbox
        names = [p["filename"] for p in st.session_state.processed]
        sel = st.selectbox("Select processed dataset", names)
        p = next(item for item in st.session_state.processed if item["filename"] == sel)
        st.subheader(f"Profile - {sel} — domain: {p['domain']}")
        prof_df = p["profile"].copy()
        # add invalid counts per column if available
        inv_map = p.get("invalids", {})
        prof_df["invalids"] = prof_df["column"].map(inv_map).fillna(0).astype(int)
        prof_df["dq_pct"] = prof_df.apply(
            lambda r: round(
                ((r["non_null"] - r["invalids"]) / r["non_null"] * 100)
                if r["non_null"] > 0
                else 100.0,
                2,
            ),
            axis=1,
        )
        # style row color by dq_pct thresholds
        def color_row(val):
            return f"color:{dq_color_pct(val)}; font-weight:bold;"
        st.dataframe(prof_df, use_container_width=True)
        # interactive per-column rule application using dropdowns
        st.markdown("### Apply / Update rules on columns")
        cols = prof_df["column"].tolist()
        sel_col = st.selectbox("Select column to assign rules", cols)
        # present available rule tokens from rules_master domain + general list
        domain_rules = rules_master.get(p["domain"], {})
        existing_list = (
            sum([v for v in domain_rules.get("rules_per_column", {}).values()], [])
            if domain_rules
            else []
        )
        global_options = list(
            set(
                existing_list
                + [
                    "not_null",
                    "email_format_check",
                    "phone_format_check",
                    "regex:",
                    "numeric_range:min:max",
                    "date_format:%Y-%m-%d",
                    "ifelse:expr",
                ]
            )
        )
        # Get existing rules for the selected column to use as default
        existing_rules_for_col = (
            rules_master.get(p["domain"], {})
            .get("rules_per_column", {})
            .get(sel_col, [])
        )
        sel_rules = st.multiselect(
            "Pick rules to apply (you can add multiple)",
            options=global_options,
            default=existing_rules_for_col,
        )
        if st.button("Apply selected rules to column"):
            # persist into rules_master under domain
            rules_master.setdefault(p["domain"], {}).setdefault("rules_per_column", {})
            # This is the key change: it overwrites the existing rules for the column.
            rules_master[p["domain"]]["rules_per_column"][sel_col] = sel_rules
            # update synonyms: map sel_col lower to canonical entry
            synonyms_master.setdefault(sel_col.lower(), [])
            if sel_col.lower() not in synonyms_master[sel_col.lower()]:
                synonyms_master[sel_col.lower()].append(sel_col.lower())
            save_yaml(RULES_FILE, rules_master)
            save_yaml(SYN_FILE, synonyms_master)
            
            # Re-run DQ checks and update session state
            for i, item in enumerate(st.session_state.processed):
                if item["filename"] == sel:
                    updated_issues, updated_matched_map, updated_invalids = apply_rules(
                        item['domain'], item['df'], rules_master, synonyms_master
                    )
                    st.session_state.processed[i]['issues'] = updated_issues
                    st.session_state.processed[i]['matched_map'] = updated_matched_map
                    st.session_state.processed[i]['invalids'] = updated_invalids
                    
                    prof_df_to_update = st.session_state.processed[i]['profile']
                    prof_df_to_update.loc[prof_df_to_update['column'] == sel_col, 'applied_rules'] = ", ".join(sel_rules)
                    st.session_state.processed[i]['profile'] = prof_df_to_update
                    break
            
            st.success(f"Applied rules to {sel_col} and updated rules.yaml & synonyms")
            st.rerun()

# -------- Rules Manager tab --------
with tabs[5]:
    st.header("Rules Manager — Guided & Advanced")
    st.markdown("Guided builder (dropdown) or Advanced (python expression). Use Test before Save.")
    domain_list = sorted(list(rules_master.keys()))
    guided_domain = st.selectbox("Domain for guided builder", domain_list, index=0)
    processed_options = [p["filename"] for p in st.session_state.processed]
    sample_file = st.selectbox("Pick processed dataset for testing", [None] + processed_options, key="guided_file_select")
    sample_col = None
    if sample_file:
        df_test = next(
            p for p in st.session_state.processed if p["filename"] == sample_file
        )["df"]
        sample_col = st.selectbox("Choose a column to test", df_test.columns.tolist(), key="guided_col_select")
    # Guided builder
    with st.expander("Guided builder (simple)"):
        guided_cond = st.selectbox(
            "Condition",
            ["not_null", "equals", "contains", "regex", "numeric_range", "date_format"],
            key="gcond",
        )
        guided_val = st.text_input(
            "Value (for equals/contains/regex/min:max/dateformat)", key="gval"
        )
        if st.button("Test guided rule"):
            if not sample_file or not sample_col:
                st.warning("Pick a processed dataset and column")
            else:
                df_sample = df_test
                if guided_cond == "not_null":
                    mask = (
                        df_sample[sample_col].isnull()
                        | (df_sample[sample_col].astype(str).str.strip() == "")
                    )
                    fails = mask.sum()
                    total = len(df_sample)
                    st.metric(
                        "Valid %",
                        f"{round(((total-fails)/total)*100,2)}",
                        delta=f"{fails} invalid",
                    )
                    if fails > 0:
                        st.dataframe(df_sample[mask].head(200))
                elif guided_cond == "equals":
                    mask = df_sample[sample_col].astype(str) != str(guided_val)
                    fails = mask.sum()
                    total = len(df_sample)
                    st.metric(
                        "Valid %",
                        f"{round(((total-fails)/total)*100,2)}",
                        delta=f"{fails} invalid",
                    )
                    if fails > 0:
                        st.dataframe(df_sample[mask].head(200))
                elif guided_cond == "contains":
                    mask = ~df_sample[sample_col].astype(str).str.contains(guided_val, na=False)
                    fails = mask.sum()
                    total = len(df_sample)
                    st.metric(
                        "Valid %",
                        f"{round(((total-fails)/total)*100,2)}",
                        delta=f"{fails} invalid",
                    )
                    if fails > 0:
                        st.dataframe(df_sample[mask].head(200))
                elif guided_cond == "regex":
                    pat = guided_val
                    mask = ~df_sample[sample_col].astype(str).str.match(pat, na=False)
                    fails = mask.sum()
                    total = len(df_sample)
                    st.metric(
                        "Valid %",
                        f"{round(((total-fails)/total)*100,2)}",
                        delta=f"{fails} invalid",
                    )
                    if fails > 0:
                        st.dataframe(df_sample[mask].head(200))
                elif guided_cond == "numeric_range":
                    try:
                        mn, mx = guided_val.split(":")
                        numeric = pd.to_numeric(df_sample[sample_col], errors="coerce")
                        mask = numeric.isnull() | (numeric < float(mn)) | (numeric > float(mx))
                        fails = mask.sum()
                        total = len(df_sample)
                        st.metric(
                            "Valid %",
                            f"{round(((total-fails)/total)*100,2)}",
                            delta=f"{fails} invalid",
                        )
                        if fails > 0:
                            st.dataframe(df_sample[mask].head(200))
                    except Exception:
                        st.error("numeric_range expects min:max")
                elif guided_cond == "date_format":
                    fmt = guided_val
                    parsed = pd.to_datetime(df_sample[sample_col], format=fmt, errors="coerce")
                    mask = parsed.isnull()
                    fails = mask.sum()
                    total = len(df_sample)
                    st.metric(
                        "Valid %",
                        f"{round(((total-fails)/total)*100,2)}",
                        delta=f"{fails} invalid",
                    )
                    if fails > 0:
                        st.dataframe(df_sample[mask].head(200))
        # Save guided rule snippet
        if st.button("Save guided rule to rules.yaml"):
            if not sample_col:
                st.warning("Pick a sample column to know where to save")
            else:
                rule_token = None
                if guided_cond == "not_null":
                    rule_token = "not_null"
                elif guided_cond == "regex":
                    rule_token = f"regex:{guided_val}"
                elif guided_cond == "numeric_range":
                    rule_token = f"numeric_range:{guided_val}"
                elif guided_cond == "date_format":
                    rule_token = f"date_format:{guided_val}"
                else:
                    # 'equals' and 'contains' are ad-hoc; convert to ifelse checking inequality (invalid when condition fails)
                    if guided_cond == "equals":
                        rule_token = f"ifelse:value != '{guided_val}'"
                    elif guided_cond == "contains":
                        # invalid when not contains
                        rule_token = f"ifelse:('{guided_val}' not in str(value))"
                if rule_token:
                    rules_master.setdefault(guided_domain, {}).setdefault(
                        "rules_per_column", {}
                    )
                    rules_master[guided_domain]["rules_per_column"].setdefault(
                        sample_col, []
                    )
                    if (
                        rule_token
                        not in rules_master[guided_domain]["rules_per_column"][sample_col]
                    ):
                        rules_master[guided_domain]["rules_per_column"][sample_col].append(
                            rule_token
                        )
                    # if regex, update domain regex map
                    if guided_cond == "regex":
                        rules_master[guided_domain].setdefault("regexes", {})[
                            sample_col
                        ] = guided_val
                    save_yaml(RULES_FILE, rules_master)
                    # update synonyms
                    synonyms_master.setdefault(sample_col.lower(), [])
                    if sample_col.lower() not in synonyms_master[sample_col.lower()]:
                        synonyms_master[sample_col.lower()].append(sample_col.lower())
                    save_yaml(SYN_FILE, synonyms_master)
                    st.success("Saved guided rule")
    st.markdown("---")
    # Advanced editor
    st.subheader("Advanced rule editor (expression)")
    adv_domain = st.selectbox("Domain (advanced)", domain_list, index=0)
    processed_options = [p["filename"] for p in st.session_state.processed]
    adv_file = st.selectbox(
        "Choose processed dataset to test (advanced)",
        [None] + processed_options,
        key="advanced_file_select"
    )
    adv_col = None
    if adv_file:
        adv_df = next(p for p in st.session_state.processed if p["filename"] == adv_file)["df"]
        adv_col = st.selectbox("Choose column (advanced)", [None] + list(adv_df.columns), key="advanced_col_select")
    st.markdown("Enter a python expression using variable value. If expression returns True → row is INVALID.")
    st.markdown("You can also type plain English like `length should be greater than 5` or `value cannot be null`.")
    adv_expr = st.text_area("Expression", value="len(str(value)) == 0", height=160)
    
    # Check for natural language phrase
    expr_to_eval = natural_language_to_expression(adv_expr)
    if expr_to_eval:
        st.info(f"Interpreted your phrase as: `{expr_to_eval}`")
        adv_expr = expr_to_eval
    
    if st.button("Test advanced expression"):
        if not adv_file or not adv_col:
            st.warning("Choose dataset and column to test")
        else:
            df_sample = adv_df
            rule_token = f"ifelse:{adv_expr}"
            tmp_rules = deepcopy(rules_master)
            tmp_rules.setdefault(adv_domain, {}).setdefault("rules_per_column", {})
            tmp_rules[adv_domain]["rules_per_column"].setdefault(adv_col, [])
            if rule_token not in tmp_rules[adv_domain]["rules_per_column"][adv_col]:
                tmp_rules[adv_domain]["rules_per_column"][adv_col].append(rule_token)
            issues_tmp, matched_tmp, inv_tmp = apply_rules(
                adv_domain,
                df_sample,
                tmp_rules,
                synonyms_master,
                selected_actual_columns=[adv_col],
            )
            fails = inv_tmp.get(adv_col, 0)
            total = len(df_sample)
            st.metric(
                "Valid %",
                f"{round(((total-fails)/total)*100,2)}",
                delta=f"{fails} invalid",
            )
            if issues_tmp:
                for k, v in issues_tmp.items():
                    st.dataframe(v.head(200))
    if st.button("Save advanced rule"):
        if not adv_col:
            st.warning("Pick a column")
        else:
            rules_master.setdefault(adv_domain, {}).setdefault("rules_per_column", {})
            token = f"ifelse:{adv_expr}"
            rules_master[adv_domain]["rules_per_column"].setdefault(adv_col, [])
            if token not in rules_master[adv_domain]["rules_per_column"][adv_col]:
                rules_master[adv_domain]["rules_per_column"][adv_col].append(token)
            save_yaml(RULES_FILE, rules_master)
            synonyms_master.setdefault(adv_col.lower(), [])
            if adv_col.lower() not in synonyms_master[adv_col.lower()]:
                synonyms_master[adv_col.lower()].append(adv_col.lower())
            save_yaml(SYN_FILE, synonyms_master)
            st.success("Saved advanced rule")

# -------- Deduplication tab --------
with tabs[6]:
    st.header("Deduplication & Merge")
    if not st.session_state.processed:
        st.info("Process datasets first (Upload & Process tab).")
    else:
        names = [p["filename"] for p in st.session_state.processed]
        pick = st.selectbox(
            "Pick processed dataset", names
        )
        pinfo = next(p for p in st.session_state.processed if p["filename"] == pick)
        df = pinfo["df"]
        st.write(f"Domain: {pinfo['domain']} — Rows: {len(df)}")
        match_cols = st.multiselect(
            "Columns to use for matching (choose 1..n)",
            df.columns.tolist(),
            default=[df.columns[0]],
        )
        thr = st.slider("Fuzzy threshold", 60, 100, 85)
        if st.button("Run Match"):
            pairs, pot = detect_potential_duplicates(df, match_cols, threshold=thr)
            if pot.empty:
                st.info("No potential duplicates detected.")
            else:
                st.markdown("Candidate duplicates")
                st.dataframe(pot)
        # fuzzy group merge section
        col_for_fuzzy = st.selectbox(
            "Also detect fuzzy groups for column (optional)", [None] + df.columns.tolist()
        )
        if col_for_fuzzy:
            groups = fuzzy_groups_for_column(df, col_for_fuzzy, threshold=thr)
            if not groups:
                st.info("No fuzzy clusters found.")
            else:
                st.markdown("Fuzzy groups")
                for i, cluster in enumerate(groups, 1):
                    st.markdown(f"Group {i}: {cluster}")
                    grp_rows = df[df[col_for_fuzzy].astype(str).isin(cluster)]
                    st.dataframe(grp_rows)
                    chosen_idx = st.radio(
                        f"Pick master index for group {i}",
                        options=grp_rows.index.tolist(),
                        key=f"master_{i}",
                    )
                    st.write(f"Selected: {chosen_idx}")
                # Manual merge across fuzzy groups: build consolidated rows
                if st.button("Manual merge fuzzy groups"):
                    consolidated = []
                    for i, cluster in enumerate(groups, 1):
                        grp_rows = df[df[col_for_fuzzy].astype(str).isin(cluster)]
                        chosen_idx = st.session_state.get(
                            f"master_{i}", grp_rows.index.tolist()[0]
                        )
                        consolidated.append(df.loc[chosen_idx])
                    final_manual = pd.concat(
                        [
                            df[
                                ~df[col_for_fuzzy]
                                .astype(str)
                                .isin(sum(groups, []))
                            ].reset_index(drop=True),
                            pd.DataFrame(consolidated).reset_index(drop=True),
                        ],
                        ignore_index=True,
                    )
                    outname = (
                        f"{pick}_fuzzy_manual_{int(datetime.datetime.now().timestamp())}.csv"
                    )
                    st.download_button(
                        "Download Manual Consolidated",
                        final_manual.to_csv(index=False).encode(),
                        file_name=outname,
                    )
                    # record merge history
                    cur = DB_CONN.cursor()
                    merge_group_id = f"{pick}_manual_fuzzy_{int(datetime.datetime.now().timestamp())}"
                    cur.execute("""
                        INSERT INTO merge_history (merge_group_id, domain, file, method, rows_merged, output_file, ts)
                        VALUES (?,?,?,?,?,?,?)
                    """, (merge_group_id, pinfo["domain"], pick, "manual_fuzzy", len(consolidated), outname, datetime.datetime.now().isoformat()))
                    merge_id = cur.lastrowid
                    # cross_reference: map consolidated rows to merged ones (best-effort: if consolidated came from chosen_idx)
                    for i, cluster in enumerate(groups, 1):
                        grp_rows = df[df[col_for_fuzzy].astype(str).isin(cluster)]
                        chosen_idx = st.session_state.get(f"master_{i}", grp_rows.index.tolist()[0])
                        survivor = df.loc[chosen_idx]
                        for idx in grp_rows.index.tolist():
                            if idx == chosen_idx:
                                continue
                            cur.execute("""
                                INSERT INTO cross_reference (merge_id, domain, file, survivor_rowid, merged_rowid, ts)
                                VALUES (?,?,?,?,?,?)
                            """, (merge_id, pinfo["domain"], pick, str(chosen_idx), str(idx), datetime.datetime.now().isoformat()))
                    DB_CONN.commit()
                    st.success("Manual fuzzy merge completed.")
        # Auto merge pairwise
        if st.button("Auto merge pairwise"):
            pairs, pot = detect_potential_duplicates(df, match_cols, threshold=thr)
            merged = []
            seen = set()
            for a, b in pairs:
                if a in seen or b in seen:
                    continue
                grp = df.loc[[a, b]]
                merged.append(merge_records_auto(grp))
                seen.update([a, b])
            final = pd.concat([df.drop(index=list(seen)), *merged], ignore_index=True)
            outname = f"{pick}_pairwise_auto_{int(datetime.datetime.now().timestamp())}.csv"
            st.download_button("Download Auto Merged", final.to_csv(index=False).encode(), file_name=outname)
            # record merge_history and cross_reference
            if DB_CONN:
                cur = DB_CONN.cursor()
                merge_group_id = f"{pick}_auto_pairwise_{int(datetime.datetime.now().timestamp())}"
                cur.execute("INSERT INTO merge_history (merge_group_id, domain, file, method, rows_merged, output_file, ts) VALUES (?,?,?,?,?,?,?)",
                            (merge_group_id, pinfo["domain"], pick, "auto_pairwise", len(merged), outname, datetime.datetime.now().isoformat()))
                merge_id = cur.lastrowid
                for grp_df in merged:
                    # best-effort mapping: assume grp_df came from two rows; we can't capture original row ids after aggregation reliably here
                    try:
                        # If grp_df has index from original, use it. Otherwise skip detailed mapping.
                        for col in grp_df.columns:
                            pass
                        # Can't recover original rowids here reliably; skip if not available
                    except Exception:
                        pass
                DB_CONN.commit()
            st.success("Auto pairwise merge completed.")
        # Manual step-through
        if st.button("Manual pairwise merge (step-through)"):
            pairs, pot = detect_potential_duplicates(df, match_cols, threshold=thr)
            st.session_state["manual_pairs"] = pairs
            st.session_state["manual_idx"] = 0
            st.rerun()
        # Manual pairwise workflow
        if st.session_state.get("manual_pairs"):
            mp = st.session_state["manual_pairs"]
            idx = st.session_state.get("manual_idx", 0)
            if idx < len(mp):
                i, j = mp[idx]
                grp = df.loc[[i, j]]
                st.write(f"Pair {idx+1}/{len(mp)} — rows {i} & {j}")
                st.dataframe(grp)
                chosen = {}
                for c in df.columns:
                    opts = grp[c].dropna().astype(str).unique().tolist()
                    if not opts:
                        opts = [""]
                    chosen[c] = st.selectbox(f"Value for {c}", opts, key=f"man_{idx}_{c}")
                if st.button("Accept & Next"):
                    st.session_state.setdefault("manual_merged_rows", []).append(pd.DataFrame([chosen]))
                    st.session_state["manual_idx"] += 1
                    st.rerun()
            else:
                # Finish manual merge
                merged_rows = st.session_state.get("manual_merged_rows", [])
                pairs = st.session_state.get("manual_pairs", [])
                seen = [i for pair in pairs for i in pair]
                final = pd.concat([df.drop(index=seen), *merged_rows], ignore_index=True)
                outname = f"{pick}_manual_pairs_{int(datetime.datetime.now().timestamp())}.csv"
                st.download_button("Download Manual Merge", final.to_csv(index=False).encode(), file_name=outname)
                if DB_CONN:
                    cur = DB_CONN.cursor()
                    cur.execute("INSERT INTO merge_history (merge_group_id, domain, file, method, rows_merged, output_file, ts) VALUES (?,?,?,?,?,?,?)",
                                (f"{pick}_manual_pairwise_{int(datetime.datetime.now().timestamp())}", pinfo["domain"], pick, "manual_pairwise", len(merged_rows), outname, datetime.datetime.now().isoformat()))
                    merge_id = cur.lastrowid
                    # store cross refs based on manual_merged_rows mapping if possible
                    # best-effort: not storing detailed mapping here
                    DB_CONN.commit()
                st.success("Manual merge finished")
                # Cleanup session state
                del st.session_state["manual_pairs"]
                del st.session_state["manual_idx"]
                if "manual_merged_rows" in st.session_state:
                    del st.session_state["manual_merged_rows"]

# -------- Merge History tab --------
with tabs[7]:
    st.header("Merge History & Cross-Reference")
    cur = DB_CONN.cursor()

    cur.execute("SELECT id, merge_group_id, domain, file, method, rows_merged, output_file, ts FROM merge_history ORDER BY ts DESC")
    rows = cur.fetchall()
    if not rows:
        st.info("No merge history available yet.")
    else:
        df_hist = pd.DataFrame(rows, columns=["id", "merge_group_id", "domain", "file", "method", "rows_merged", "output_file", "ts"])
        st.dataframe(df_hist, use_container_width=True)

        sel_merge = st.selectbox("Select merge event to inspect", [None] + df_hist["merge_group_id"].tolist())
        if sel_merge:
            mid = df_hist.loc[df_hist["merge_group_id"] == sel_merge, "id"].values[0]
            cur.execute("SELECT id, survivor_rowid, merged_rowid, ts FROM cross_reference WHERE merge_id = ?", (mid,))
            ref_rows = cur.fetchall()
            if not ref_rows:
                st.info("No cross-reference records found.")
            else:
                df_ref = pd.DataFrame(ref_rows, columns=["xref_id", "Survivor", "Merged", "Timestamp"])
                st.subheader("Cross Reference Records")
                st.dataframe(df_ref, use_container_width=True)

                # Allow unmerge selected records
                st.markdown("Select cross-reference rows to unmerge (delete mapping)")
                to_unmerge = st.multiselect("Select xref_id to delete", df_ref["xref_id"].tolist())
                if to_unmerge and st.button("🔄 Unmerge Selected Records"):
                    try:
                        for x in to_unmerge:
                            cur.execute("DELETE FROM cross_reference WHERE id = ?", (x,))
                        DB_CONN.commit()
                        st.success(f"Successfully unmerged {len(to_unmerge)} mapping(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Unmerge failed: {e}")

                if st.button("🔄 Unmerge Entire Merge Event"):
                    try:
                        cur.execute("DELETE FROM cross_reference WHERE merge_id = ?", (mid,))
                        cur.execute("DELETE FROM merge_history WHERE id = ?", (mid,))
                        DB_CONN.commit()
                        st.success(f"Successfully unmerged merge group {sel_merge}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Unmerge failed: {e}")

# -------- Trends tab --------
with tabs[8]:
    st.header("Trends & Monitoring")
    if DB_CONN:
        cur = DB_CONN.cursor()
        domain_list = sorted(list(rules_master.keys())) + ["unknown"]
        sel_domain = st.selectbox("Filter domain", ["All"] + domain_list, index=0)
        q = "SELECT ts, domain, file, column_name, invalids, rows, valid_pct FROM trends"
        if sel_domain != "All":
            q += " WHERE domain = ?"
            cur.execute(q, (sel_domain,))
        else:
            cur.execute(q)
        rows = cur.fetchall()
        if not rows:
            st.info("No trend data available yet.")
        else:
            df_tr = pd.DataFrame(rows, columns=["ts", "domain", "file", "column", "invalids", "rows", "valid_pct"])
            df_tr["ts"] = pd.to_datetime(df_tr["ts"], errors="coerce")
            st.dataframe(df_tr.sort_values("ts", ascending=False).reset_index(drop=True), use_container_width=True)
            # Plot
            if df_tr["valid_pct"].notna().any():
                columns = sorted(df_tr["column"].dropna().unique().tolist())
                sel_cols = st.multiselect("Select columns to plot", columns, default=columns[:3] if columns else [])
                plot_df = df_tr[df_tr["column"].isin(sel_cols)] if sel_cols else df_tr
                if not plot_df.empty:
                    fig = px.line(plot_df, x="ts", y="valid_pct", color="column", markers=True, title="DQ % over time")
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            # Export CSV
            csv_bytes = df_tr.to_csv(index=False).encode()
            st.download_button("Download Trend CSV", csv_bytes, file_name="dq_trend.csv")
    # Final save of configs
    save_yaml(RULES_FILE, rules_master)
    save_yaml(SYN_FILE, synonyms_master)
    save_yaml(PDO_FILE, pdo_master)
