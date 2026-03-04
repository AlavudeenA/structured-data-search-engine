# File: streamlit_app.py
"""Streamlit UI for hybrid structured + analytical workflow."""

from __future__ import annotations

import json

import streamlit as st

from src.orchestrator import handle_user_query
from src.sql_autofix import ingest_sql_with_autofix
from src.sql_executor import execute_sql
from src.vector_store import (
    DEFAULT_COLLECTION,
    delete_capsule_by_id,
    list_capsules,
    reset_collection,
)


def _compute_analytical_confidence(hits: list[dict]) -> str:
    if not hits:
        return "Low"
    top_score = float(hits[0].get("score", 0.0))
    if top_score >= 0.75:
        return "High"
    if top_score >= 0.5:
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
st.title("Compliance Data Assistant")
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ask Question", "Ingest SQL to Vector DB", "Manage Capsules", "Reset Vector DB"]
)

with tab1:
    st.subheader("Ask in English")
    user_question = st.text_area(
        "Question",
        placeholder="Example: Show top broker dealers by trade count",
        height=120,
    )
    run_question = st.button("Run Question", type="primary")
    with st.expander("Analytical retrieval filters (optional)"):
        retrieval_capsule_type = st.selectbox(
            "Capsule type filter",
            options=["", "aggregation", "trend", "correlation", "anomaly", "entity_profile"],
            index=0,
        )
        retrieval_entity = st.selectbox(
            "Entity filter",
            options=["", "BrokerDealer", "Employee", "Account", "TradeRequest", "Security"],
            index=0,
        )
        retrieval_topic = st.selectbox(
            "Capsule topic filter",
            options=["", "broker_activity", "employee_activity", "security_trend", "account_activity"],
            index=0,
        )

    if run_question:
        if not user_question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running workflow..."):
                result = handle_user_query(
                    user_question.strip(),
                    capsule_type_filter=retrieval_capsule_type or None,
                    entity_filter=retrieval_entity or None,
                    capsule_topic_filter=retrieval_topic or None,
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
    st.subheader("Ingest Query Result as Context Capsules")
    sql_text = st.text_area(
        "SQL Query (SELECT only)",
        placeholder="SELECT TOP 100 * FROM TradeRequest ORDER BY RequestDate DESC",
        height=140,
    )
    batch_size = st.number_input("Capsule batch size", min_value=5, max_value=200, value=25)
    capsule_name = st.text_input(
        "Capsule Name (optional)",
        placeholder="broker_activity_summary",
    )
    capsule_type = st.selectbox(
        "Capsule Type",
        options=["aggregation", "trend", "correlation", "anomaly", "entity_profile"],
        index=0,
    )
    entity = st.selectbox(
        "Entity",
        options=["", "BrokerDealer", "Employee", "Account", "TradeRequest", "Security"],
        index=0,
    )
    capsule_topic = st.text_input(
        "Capsule Topic (optional)",
        placeholder="broker_activity",
    )
    capsule_priority = st.selectbox(
        "Capsule Priority",
        options=["", "high", "medium", "low"],
        index=0,
        help="If empty, priority is auto-derived from capsule type.",
    )
    metric_tags_input = st.text_input(
        "Metric Tags (comma-separated, optional)",
        placeholder="trade_request_count,monthly_volume",
    )
    ingestion_mode = st.selectbox(
        "Ingestion mode",
        options=["append_unique", "replace_source"],
        index=1,
        help="append_unique keeps existing sources and upserts by stable content hash. "
        "replace_source deletes previous capsules for this SQL source before inserting.",
    )
    run_ingest = st.button("Run Ingestion", type="primary")

    if run_ingest:
        if not sql_text.strip():
            st.warning("Enter SQL first.")
        else:
            try:
                with st.spinner("Executing SQL, auto-fixing if needed, and indexing capsules..."):
                    outcome = ingest_sql_with_autofix(
                        sql=sql_text.strip(),
                        source_query="manual_sql_ingestion",
                        batch_size=int(batch_size),
                        ingestion_mode=ingestion_mode,
                        capsule_metadata={
                            "capsule_name": capsule_name.strip() or None,
                            "capsule_type": capsule_type,
                            "entity": entity,
                            "capsule_topic": capsule_topic.strip(),
                            "capsule_priority": capsule_priority or "",
                            "metric_tags": [
                                tag.strip()
                                for tag in metric_tags_input.split(",")
                                if tag.strip()
                            ],
                        },
                    )
                ingest_result = outcome["ingest_result"]
                st.success("Ingestion complete.")
                st.json(ingest_result)
                if ingestion_mode == "replace_source":
                    st.info(
                        f"Old capsules removed for this source: {ingest_result.get('capsules_deleted', 0)}"
                    )

                if outcome.get("corrected"):
                    st.info("SQL had issues and was auto-corrected using Groq.")
                    st.write("Original SQL:")
                    st.code(outcome.get("original_sql", ""), language="sql")
                    st.write("Corrected SQL:")
                    st.code(outcome.get("final_sql", ""), language="sql")
                    with st.expander("Auto-fix attempts"):
                        st.json(outcome.get("attempts", []))

                with st.expander("Preview SQL result rows"):
                    execution = execute_sql(outcome.get("final_sql", sql_text.strip()))
                    st.write(f"Row count: {execution.get('row_count', 0)}")
                    rows = execution.get("rows", [])
                    if rows:
                        st.dataframe(rows)
                    else:
                        st.info("No rows returned.")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

with tab3:
    st.subheader("All Capsules")
    view_collection = st.text_input(
        "Collection name for viewing",
        value=DEFAULT_COLLECTION,
        key="view_collection_name",
    )
    max_items = st.number_input("Max capsules to load", min_value=10, max_value=5000, value=500)
    load_capsules = st.button("Load Capsules", key="load_capsules_btn")

    if "view_capsules" not in st.session_state:
        st.session_state["view_capsules"] = []

    if load_capsules:
        st.session_state["view_capsules"] = list_capsules(
            collection_name=view_collection.strip(),
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
                collection_name=view_collection.strip(),
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
with tab4:
    st.subheader("Reset Embedded Context Store")
    st.warning("This will delete all embedded capsules in the selected collection.")
    collection_name = st.text_input("Collection name", value=DEFAULT_COLLECTION)
    confirm_reset = st.checkbox("I understand this action will remove stored vectors.")
    run_reset = st.button("Reset Collection", key="reset_collection_btn")

    if run_reset:
        if not confirm_reset:
            st.error("Please confirm reset by checking the confirmation box.")
        else:
            result = reset_collection(collection_name=collection_name.strip())
            st.success("Collection reset complete.")
            st.json(result)

st.divider()
st.caption(
    "Tip: First ingest representative SQL outputs in Tab 2. "
    "Then ask analytical questions in Tab 1 to retrieve relevant capsules."
)
