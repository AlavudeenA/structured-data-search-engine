# File: streamlit_app.py
"""Streamlit UI for hybrid structured + analytical workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from src.app_constants import (
    CAPSULE_TYPE_AGGREGATION,
    CAPSULE_TYPE_ANOMALY,
    CAPSULE_TYPE_DISTRIBUTION,
    CAPSULE_TYPE_RANDOM_SAMPLE,
    CAPSULE_TYPE_SUMMARY,
    CAPSULE_TYPE_TREND,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    DEFAULT_MAX_GROUP_COLS_PER_TABLE,
    DEFAULT_MAX_RANDOM_PER_TABLE,
    DEFAULT_ROWS_PER_CAPSULE,
    DEFAULT_TARGET_CAPSULES,
    GEN_ROWS_MAX_ALLOWED,
    GEN_ROWS_MAX_EXCLUSIVE,
    GEN_ROWS_MIN,
    INGESTION_MODE_APPEND_UNIQUE,
    MANUAL_ROWS_MAX,
    MAX_GROUP_COLS_MAX,
    MAX_GROUP_COLS_MIN,
    TARGET_CAPSULES_MAX,
    TARGET_CAPSULES_MIN,
    UI_VIEW_DEFAULT_ITEMS,
    UI_VIEW_MAX_ITEMS,
    UI_VIEW_MIN_ITEMS,
)
from src.database_connection import execute_select
from src.capsule_generator import preview_capsule_sql_plans
from src.embedding import generate_and_ingest_capsules, ingest_capsules
from src.orchestrator import handle_user_query
from src.vector_store import (
    DEFAULT_COLLECTION,
    delete_capsule_by_id,
    list_capsules,
    purge_local_qdrant_storage,
    reset_all_collections,
)


def _compute_analytical_confidence(hits: list[dict]) -> str:
    if not hits:
        return "Low"
    top_score = float(hits[0].get("score", 0.0))
    if top_score >= CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _supporting_capsule_name(hits: list[dict], supporting: dict | None = None) -> str:
    supporting = supporting or {}
    mode = str(supporting.get("mode", "")).strip()
    names = supporting.get("capsule_names", []) or []
    if mode == "grouped_by_source_hash" and names:
        count = int(supporting.get("capsule_count", len(names)))
        if len(names) == 1:
            return f"{names[0]} (grouped source, capsules={count})"
        preview = ", ".join(names[:2])
        more = f", +{len(names) - 2} more" if len(names) > 2 else ""
        return f"{preview}{more} (grouped source, capsules={count})"
    if mode == "single" and names:
        return names[0]
    if not hits:
        return "N/A"
    payload = hits[0].get("payload", {}) or {}
    return str(payload.get("capsule_name", "N/A"))


def _capsule_info_rows(hits: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for hit in hits[:5]:
        payload = hit.get("payload", {}) or {}
        rows.append(
            {
                "capsule_name": payload.get("capsule_name", ""),
                "type": payload.get("capsule_type", ""),
                "entity": payload.get("entity", ""),
                "topic": payload.get("capsule_topic", ""),
                "priority": payload.get("capsule_priority", ""),
                "score": round(float(hit.get("score", 0.0)), 3),
                "row_count": payload.get("row_count", ""),
                "source_sql_hash": str(payload.get("source_sql_hash", ""))[:12],
            }
        )
    return rows


st.set_page_config(page_title="Compliance Hybrid Query UI", layout="wide")
st.title("Structrual Data Assistant")
st.caption("Structured queries -> Text-to-SQL | Analytical queries -> Vector Retrieval")
st.markdown(
    """
    <style>
    .st-key-delete_selected_capsule_btn button {
        border: 2px solid #b00020 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    .st-key-reset_collection_btn button {
        border: 2px solid #b00020 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    .st-key-load_capsules_btn button {
        border: 2px solid #1f7a1f !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    .st-key-load_capsules_btn button:hover {
        border-color: #155d15 !important;
        box-shadow: none !important;
    }
    .st-key-load_capsules_btn button:focus,
    .st-key-load_capsules_btn button:focus-visible {
        border-color: #155d15 !important;
        box-shadow: 0 0 0 1px rgba(31, 122, 31, 0.25) !important;
        outline: none !important;
    }
    .st-key-delete_selected_capsule_btn button:hover {
        border-color: #8c0018 !important;
        box-shadow: none !important;
    }
    .st-key-reset_collection_btn button:hover {
        border-color: #8c0018 !important;
        box-shadow: none !important;
    }
    .st-key-delete_selected_capsule_btn button:focus,
    .st-key-delete_selected_capsule_btn button:focus-visible {
        border-color: #8c0018 !important;
        box-shadow: 0 0 0 1px rgba(176, 0, 32, 0.25) !important;
        outline: none !important;
    }
    .st-key-reset_collection_btn button:focus,
    .st-key-reset_collection_btn button:focus-visible {
        border-color: #8c0018 !important;
        box-shadow: 0 0 0 1px rgba(176, 0, 32, 0.25) !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Ask Question",
        "Generate Capsules ",
        "Insert Capsule",
        "Manage Capsules",
        "Reset Vector DB",
    ]
)

with tab1:
    st.subheader("Ask in English")
    user_question = st.text_area(
        "Question",
        placeholder="Example: Show top broker dealers by trade count",
        height=120,
    )
    force_capsule_retrieval = st.checkbox(
        "Force capsule retrieval (skip Text-to-SQL)",
        value=False,
        help="Useful when you want answers only from vector-retrieved capsule context.",
    )
    run_question = st.button("Run Question", type="primary")

    if run_question:
        if not user_question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running workflow..."):
                result = handle_user_query(
                    user_question.strip(),
                    force_analytical=force_capsule_retrieval,
                )

            st.success(f"Route: {result.get('route')}")
            st.write("Intent:", result.get("intent", {}))
            st.write("Answer:")
            st.code(result.get("answer", ""), language="text")

            if result.get("route") == "text_to_sql":
                execution = result.get("execution", {})
                st.write("Generated SQL:")
                st.code(execution.get("sql", ""), language="sql")
                st.write(f"Rows: {execution.get('row_count', 0)}")
                rows = execution.get("rows", [])
                if rows:
                    st.dataframe(rows)
            else:
                retrieval = result.get("retrieval", {}) or {}
                hits = retrieval.get("hits", [])
                supporting = retrieval.get("supporting", {})
                confidence = _compute_analytical_confidence(hits)
                st.write(f"Confidence: {confidence}")
                st.write("Supporting capsule name:", _supporting_capsule_name(hits, supporting))
                capsule_rows = _capsule_info_rows(hits)
                if capsule_rows:
                    st.write("Supporting capsules:")
                    st.dataframe(capsule_rows)

with tab2:
    st.subheader("Generate and Ingest Context Capsules")
    st.caption(
        "Primary mode: schema-agnostic generation of random, aggregation, and distribution capsules. "
    )
    st.session_state["latest_planned_sqls"] = []
    gen_col1, gen_col2 = st.columns(2)
    with gen_col1:
        target_capsules = st.number_input(
            "Target capsules (count)",
            min_value=TARGET_CAPSULES_MIN,
            max_value=TARGET_CAPSULES_MAX,
            value=DEFAULT_TARGET_CAPSULES,
        )
        rows_per_capsule = st.number_input(
            "Rows per capsule (must be < 100)",
            min_value=GEN_ROWS_MIN,
            max_value=GEN_ROWS_MAX_EXCLUSIVE,
            value=DEFAULT_ROWS_PER_CAPSULE,
            step=1,
            key="rows_per_capsule_input",
        )

    with gen_col2:
        use_llm_summaries = st.checkbox(
            "Use LLM summaries (higher cost)",
            value=False,
        )
        max_random_per_table = st.number_input(
            "Max random capsules per table",
            min_value=GEN_ROWS_MIN,
            max_value=1000,
            value=DEFAULT_MAX_RANDOM_PER_TABLE,
            help=f"Retention policy: keep only latest N {CAPSULE_TYPE_RANDOM_SAMPLE} capsules per table.",
        )
        max_group_cols = st.number_input(
            "Max group columns per table",
            min_value=MAX_GROUP_COLS_MIN,
            max_value=MAX_GROUP_COLS_MAX,
            value=DEFAULT_MAX_GROUP_COLS_PER_TABLE,
        )

    rows_value = int(rows_per_capsule)
    run_generate = st.button(
        "Generate Capsules",
        type="primary",
        key="run_generate_capsules_",
    )

    if run_generate:
        if rows_value >= GEN_ROWS_MAX_EXCLUSIVE or rows_value < GEN_ROWS_MIN:
            st.error(
                f"Rows per capsule must be between {GEN_ROWS_MIN} and {GEN_ROWS_MAX_ALLOWED}. "
                "Generation was not started."
            )
            st.session_state["latest_planned_sqls"] = []
        else:
            try:
                st.session_state["latest_planned_sqls"] = preview_capsule_sql_plans(
                    target_rows=rows_value,
                    max_rows_per_capsule=rows_value,
                    include_temporal_aggregations=True,
                    max_group_cols_per_table=int(max_group_cols),
                )
                with st.spinner("Generating  capsules and indexing vectors..."):
                    result = generate_and_ingest_capsules(
                        collection_name=DEFAULT_COLLECTION,
                        target_capsules=int(target_capsules),
                        target_rows=rows_value,
                        max_rows_per_capsule=rows_value,
                        include_temporal_aggregations=True,
                        max_group_cols_per_table=int(max_group_cols),
                        use_llm_summaries=use_llm_summaries,
                        ingestion_mode=INGESTION_MODE_APPEND_UNIQUE,
                        max_random_per_table=int(max_random_per_table),
                        replace_similar_capsules=True,
                    )
                result["effective_rows_per_capsule"] = rows_value
                result["effective_target_capsules"] = int(target_capsules)
                st.success("Capsule generation and ingestion complete.")
                st.json(result)
                if use_llm_summaries:
                    st.info("LLM summarization was enabled for capsule summaries.")
            except Exception as exc:
                st.error(f" generation failed: {exc}")
                st.session_state["latest_planned_sqls"] = []

    latest_plans = st.session_state.get("latest_planned_sqls", [])
    with st.expander(f"Planned SQL queries ({len(latest_plans)})", expanded=False):
        st.code(
            "\n".join(f"{i + 1}. {sql}" for i, sql in enumerate(latest_plans)),
            language="sql",
        )

with tab3:
    st.subheader("Manual Context Capsule Insert")
    st.caption("Run your own SQL and insert a single capsule with custom metadata.")

    manual_sql = st.text_area(
        "SQL query (SELECT/WITH only, must return <= 100 rows)",
        height=140,
        key="manual_capsule_sql",
    )
    manual_capsule_type = st.selectbox(
        "Capsule type",
        options=[
            CAPSULE_TYPE_RANDOM_SAMPLE,
            CAPSULE_TYPE_AGGREGATION,
            CAPSULE_TYPE_DISTRIBUTION,
            CAPSULE_TYPE_TREND,
            CAPSULE_TYPE_ANOMALY,
            CAPSULE_TYPE_SUMMARY,
        ],
        index=5,
        key="manual_capsule_type",
    )
    manual_summary = st.text_area(
        "Summary text (optional)",
        height=90,
        key="manual_summary_text",
    )

    run_manual_insert = st.button("Insert Manual Capsule", type="primary", key="insert_manual_capsule_btn")
    if run_manual_insert:
        try:
            sql_text = manual_sql.strip()
            if not sql_text:
                raise ValueError("Please enter SQL query text.")

            query_result = execute_select(sql_text, max_rows=101)
            row_count = int(query_result.get("row_count", 0))
            if row_count > MANUAL_ROWS_MAX:
                raise ValueError(
                    f"Manual query returned {row_count} rows. Capsule row limit is {MANUAL_ROWS_MAX} (no truncation)."
                )
            rows = query_result.get("rows", [])
            if not rows:
                raise ValueError("Manual query returned no rows; nothing to ingest.")

            columns = list((query_result.get("columns") or []))

            summary = manual_summary.strip() or (
                f"Manual {manual_capsule_type} capsule from custom SQL with {row_count} rows. "
                f"Fields: {', '.join(columns[:6])}."
            )
            created_at = datetime.now(timezone.utc).isoformat()
            capsule = {
                "capsule_id": "",
                "capsule_name": "",
                "capsule_type": manual_capsule_type,
                "capsule_version": "",
                "tables_used": [],
                "key_columns": [],
                "tags": [],
                "summary_text": summary,
                "rows_json": json.dumps(rows, default=str),
                "row_count": row_count,
                "created_at": created_at,
                "metrics": {"row_count": row_count},
                "source_sql": sql_text,
            }
            ingest_result = ingest_capsules(
                capsules=[capsule],
                collection_name=DEFAULT_COLLECTION,
                ingestion_mode=INGESTION_MODE_APPEND_UNIQUE,
            )
            st.success("Manual capsule inserted.")
            st.json(ingest_result)
        except Exception as exc:
            st.error(f"Manual capsule insert failed: {exc}")

with tab4:
    st.subheader("All Context Capsules")
    max_items = st.number_input(
        "Max capsules to load",
        min_value=UI_VIEW_MIN_ITEMS,
        max_value=UI_VIEW_MAX_ITEMS,
        value=UI_VIEW_DEFAULT_ITEMS,
    )
    load_capsules = st.button("Load Capsules", key="load_capsules_btn")

    if "view_capsules" not in st.session_state:
        st.session_state["view_capsules"] = []

    if load_capsules:
        st.session_state["view_capsules"] = list_capsules(
            collection_name=DEFAULT_COLLECTION,
            limit=int(max_items),
        )

    capsules = st.session_state.get("view_capsules", [])
    st.write(f"Capsules loaded: {len(capsules)}")
    if not capsules:
        st.info("No capsules found for this collection.")
    else:
        rows = []
        for item in capsules:
            payload = item.get("payload", {})
            rows.append(
                {
                    "id": item.get("id"),
                    "capsule_name": payload.get("capsule_name", ""),
                    "type": payload.get("capsule_type", ""),
                    "entity": payload.get("entity", ""),
                    "topic": payload.get("capsule_topic", ""),
                    "priority": payload.get("capsule_priority", ""),
                    "row_count": payload.get("row_count", ""),
                    "refreshed_at_utc": payload.get("refreshed_at_utc", ""),
                    "source_sql_hash": str(payload.get("source_sql_hash", ""))[:12],
                }
            )
        st.dataframe(rows)

        options = [
            f"{item.get('id')} | {item.get('payload', {}).get('capsule_name', '')}"
            for item in capsules
        ]
        selected = st.selectbox("Select capsule to delete", options=options)
        if st.button("Delete Selected Capsule", key="delete_selected_capsule_btn"):
            selected_id = selected.split(" | ", 1)[0]
            capsule_id: object = int(selected_id) if selected_id.isdigit() else selected_id
            ok = delete_capsule_by_id(
                capsule_id=capsule_id,
                collection_name=DEFAULT_COLLECTION,
            )
            if ok:
                st.success(f"Deleted capsule id {selected_id}")
                st.session_state["view_capsules"] = [
                    c for c in capsules if str(c.get("id")) != selected_id
                ]
            else:
                st.error("Delete failed.")

        with st.expander("Full capsule payloads"):
            st.json(capsules)

st.divider()
with tab5:
    st.subheader("Reset Embedded Context Store")
    st.warning("This will delete All embedded capsules in local Qdrant storage.")
    confirm_reset = st.checkbox("I understand this action will remove stored vectors.")
    run_reset = st.button("Reset All Collections", key="reset_collection_btn")

    if run_reset:
        if not confirm_reset:
            st.error("Please confirm reset by checking the confirmation box.")
        else:
            purge_result = purge_local_qdrant_storage()
            result = reset_all_collections()
            st.session_state["view_capsules"] = []
            st.success("Vector DB fully cleared.")
            st.json({"purge": purge_result, "reset_all": result})

st.divider()
st.caption(
    "Tip: First ingest representative SQL outputs in Tab 2. "
    "Then ask analytical questions in Tab 1 to retrieve relevant capsules."
)
