"""Streamlit UI entry point for the compliance capsule engine."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.app_constants import (
    COLLECTION_ANALYTICAL,
    COLLECTION_DERIVED,
    COLLECTION_SCHEMA,
    INTENT_ANALYTICAL,
    INTENT_HYBRID,
    INTENT_OPERATIONAL,
    INTENT_STRUCTURED,
    UI_MAX_HISTORY,
)
from src.pipeline1.relationship_builder import load_graph
from src.pipeline1.store_manager import collection_stats, generate_all_capsule_collections, refresh_data_only, schema_refresh
from src.pipeline2.orchestrator import handle_query
from src.vector_store import clear_collection, collection_counts, delete_by_capsule_id, reset_all_collections, scroll_all, upsert_capsule
from src.database_connection import execute_select
from src.embedding import embed_single

st.set_page_config(page_title="Compliance Engine", page_icon="C", layout="wide")

INTENT_BADGE = {
    INTENT_STRUCTURED: ("green", "Structured"),
    INTENT_ANALYTICAL: ("blue", "Analytical"),
    INTENT_HYBRID: ("orange", "Hybrid"),
    INTENT_OPERATIONAL: ("red", "Operational"),
}


def init_state() -> None:
    defaults = {
        "question_history": [],
        "telemetry_log": [],
        "last_result": None,
        "explorer_capsules": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def render_intent_badge(intent: str, confidence: float) -> None:
    color, label = INTENT_BADGE.get(intent, ("gray", intent.title()))
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px;font-size:0.9rem;'>"
        f"{label} ({confidence:.0%})</span>",
        unsafe_allow_html=True,
    )


def record_telemetry(result) -> None:
    row = {
        "question": st.session_state.get("last_question", ""),
        "intent": result.intent,
        "route": result.route_taken,
        "capsules_used": ", ".join(result.capsules_used),
        "confidence": round(result.confidence, 3),
        "answer_ms": result.answer_ms,
        "sql_executed": bool(result.sql_generated),
        "autofix": result.autofix_used,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state.telemetry_log.append(row)


with st.sidebar:
    st.markdown("### Collections")
    st.json(collection_counts())

st.title("Compliance Engine")
st.caption("Groq + fastembed + Qdrant + SQL Server")

ask_tab, generate_tab, explorer_tab, graph_tab, telemetry_tab, insert_tab, reset_tab = st.tabs(
    [
        "Ask Question",
        "Generate Capsules",
        "Capsule Explorer",
        "Capsule Graph",
        "Telemetry",
        "Insert Capsule",
        "Reset",
    ]
)

with ask_tab:
    st.subheader("Ask Question")
    question = st.text_area("Question", height=120, placeholder="Which broker-dealer's trading activity is increasing over time?")
    history = [item["question"] for item in st.session_state.question_history[-UI_MAX_HISTORY:]]
    if history:
        selected = st.selectbox("Recent Questions", [""] + history)
        if selected and not question:
            question = selected

    if st.button("Run Question", type="primary") and question.strip():
        st.session_state.last_question = question.strip()
        with st.spinner("Running compliance pipeline..."):
            try:
                result = handle_query(question.strip())
                st.session_state.last_result = result
                st.session_state.question_history.append({"question": question.strip()})
                st.session_state.question_history = st.session_state.question_history[-UI_MAX_HISTORY:]
                record_telemetry(result)
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.last_result
    if result:
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("**Intent detected**")
            render_intent_badge(result.intent, result.confidence)
        with col2:
            st.markdown(f"**Route taken**  ` {result.route_taken} `")

        st.markdown("### Answer")
        st.write(result.answer)

        if result.sql_reason:
            st.markdown("**SQL Reason**")
            st.write(result.sql_reason)

        if result.autofix_used:
            st.info("SQL autofix was used after the initial execution failed.")
        if result.error:
            st.error(result.error)

        with st.expander("Capsules Used", expanded=False):
            if result.capsules_used:
                st.write(result.capsules_used)
            else:
                st.write("No capsules were used directly.")

        with st.expander("Context Package", expanded=False):
            st.json(result.context_package or {})

        with st.expander("Generated SQL", expanded=False):
            st.code(result.sql_generated or "", language="sql")

        with st.expander("SQL Rows Returned", expanded=False):
            if result.sql_rows:
                st.dataframe(result.sql_rows, use_container_width=True)
            else:
                st.write("No SQL rows returned.")

with generate_tab:
    st.subheader("Generate Capsules")
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if col1.button("Generate All Capsules", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting full build")
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str) -> None:
            progress_rows.append({"capsule_id": capsule_id, "status": status, "signal_preview": preview})
            total = max(len(progress_rows), 1)
            progress.progress(min(total / 32, 1.0), text=f"Built {capsule_id}")

        with st.spinner("Building all collections..."):
            summary = generate_all_capsule_collections(progress_callback=on_progress)
        progress.progress(1.0, text="Completed")
        st.success(
            f"Generated {summary.analytical_count} analytical, {summary.schema_count} schema_context, and {summary.derived_count} derived capsules."
        )
        st.write(f"Relationship graph edges: {summary.graph_edge_count}")
        st.dataframe(progress_rows, use_container_width=True)

    if col2.button("Refresh Data", type="primary", use_container_width=True):
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str) -> None:
            progress_rows.append({"capsule_id": capsule_id, "status": status, "signal_preview": preview})

        with st.spinner("Refreshing analytical capsules from saved plan..."):
            summary = refresh_data_only(progress_callback=on_progress)
        st.success(f"Refreshed {summary.analytical_count} analytical capsules. Schema-context and derived capsules were kept.")
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

    if col3.button("Schema Refresh", type="primary", use_container_width=True):
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str) -> None:
            progress_rows.append({"capsule_id": capsule_id, "status": status, "signal_preview": preview})

        with st.spinner("Refreshing schema and rebuilding all collections..."):
            summary = schema_refresh(progress_callback=on_progress)
        st.success(
            f"Schema refresh complete. Schema changed: {summary.schema_changed}. Rebuilt {summary.analytical_count} analytical, {summary.schema_count} schema_context, and {summary.derived_count} derived capsules."
        )
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

with explorer_tab:
    st.subheader("Capsule Explorer")
    if st.button("Load All Capsules"):
        analytical = scroll_all(COLLECTION_ANALYTICAL)
        schema = scroll_all(COLLECTION_SCHEMA)
        derived = scroll_all(COLLECTION_DERIVED)
        st.session_state.explorer_capsules = analytical + schema + derived

    capsules = st.session_state.explorer_capsules
    if capsules:
        type_options = sorted({capsule.get("capsule_type", "schema_context") for capsule in capsules})
        priority_options = sorted({capsule.get("priority", "") for capsule in capsules if capsule.get("priority")})
        tag_options = sorted({tag for capsule in capsules for tag in capsule.get("tags", [])})

        col1, col2, col3, col4, col5 = st.columns(5)
        filter_type = col1.multiselect("Type", type_options)
        filter_priority = col2.multiselect("Priority", priority_options)
        filter_tags = col3.multiselect("Tags", tag_options)
        filter_stale = col4.checkbox("Stale only")
        filter_anomaly = col5.checkbox("Anomaly only")

        filtered = capsules
        if filter_type:
            filtered = [capsule for capsule in filtered if capsule.get("capsule_type", "schema_context") in filter_type]
        if filter_priority:
            filtered = [capsule for capsule in filtered if capsule.get("priority") in filter_priority]
        if filter_tags:
            filtered = [capsule for capsule in filtered if any(tag in capsule.get("tags", []) for tag in filter_tags)]
        if filter_stale:
            filtered = [capsule for capsule in filtered if capsule.get("is_stale")]
        if filter_anomaly:
            filtered = [capsule for capsule in filtered if float(capsule.get("anomaly_score", 0)) > 0.7]

        st.dataframe(
            [
                {
                    "capsule_id": capsule.get("capsule_id"),
                    "type": capsule.get("capsule_type", "schema_context"),
                    "priority": capsule.get("priority", "-"),
                    "ttl_hours": capsule.get("ttl_hours", "-"),
                    "anomaly_score": capsule.get("anomaly_score", "-"),
                    "trend_direction": capsule.get("trend_direction", "-"),
                    "related_count": len(capsule.get("related_capsule_ids", [])),
                    "expires_at": capsule.get("expires_at", "-"),
                }
                for capsule in filtered
            ],
            use_container_width=True,
        )

        selected_capsule = st.selectbox("Inspect Capsule", [""] + [capsule.get("capsule_id") for capsule in filtered])
        if selected_capsule:
            capsule = next(capsule for capsule in filtered if capsule.get("capsule_id") == selected_capsule)
            st.write("**Signal / Summary**")
            st.write(capsule.get("signal") or capsule.get("summary"))
            st.write("**Embed Text**")
            st.write(capsule.get("embed_text", "-"))
            st.write("**Related Capsules**")
            st.write(list(zip(capsule.get("related_capsule_ids", []), capsule.get("relationship_types", []))))
            st.write("**Source SQL**")
            st.code(capsule.get("sql", capsule.get("sql_template", "")), language="sql")

            delete_collection = COLLECTION_ANALYTICAL
            if "summary" in capsule and "tables" in capsule:
                delete_collection = COLLECTION_SCHEMA
            elif "derived_from" in capsule:
                delete_collection = COLLECTION_DERIVED
            if st.button(f"Delete {selected_capsule}"):
                delete_by_capsule_id(delete_collection, selected_capsule)
                st.success(f"Deleted {selected_capsule}")
                st.session_state.explorer_capsules = []
                st.rerun()
    else:
        st.info("Load capsules to explore them.")

with graph_tab:
    st.subheader("Capsule Graph")
    graph = load_graph()
    if graph:
        st.write(f"Built at: {graph.built_at}")
        st.dataframe(
            [
                {
                    "capsule_id": edge.from_id,
                    "related_to": edge.to_id,
                    "relationship": edge.relationship,
                    "join_key": edge.join_key or "",
                }
                for edge in graph.edges
            ],
            use_container_width=True,
        )
        st.write("### Derived Capsules")
        st.dataframe(scroll_all(COLLECTION_DERIVED), use_container_width=True)
    else:
        st.info("No capsule graph found yet. Generate capsules first.")

with telemetry_tab:
    st.subheader("Telemetry")
    if st.session_state.telemetry_log:
        st.dataframe(st.session_state.telemetry_log, use_container_width=True)
        if st.button("Clear Log"):
            st.session_state.telemetry_log = []
            st.rerun()
    else:
        st.info("No telemetry has been recorded in this session.")

with insert_tab:
    st.subheader("Insert Capsule")
    capsule_id = st.text_input("Capsule ID")
    capsule_type = st.text_input("Capsule Type", value="aggregation")
    sql_text = st.text_area("SQL", height=180)
    summary_text = st.text_area("Summary Text", height=120)
    if st.button("Insert Capsule", type="primary"):
        if not capsule_id or not sql_text or not summary_text:
            st.error("All fields are required.")
        else:
            with st.spinner("Executing SQL and inserting capsule..."):
                rows = execute_select(sql_text, max_rows=50)
                payload = {
                    "capsule_id": capsule_id,
                    "capsule_type": capsule_type,
                    "priority": "P3",
                    "what": capsule_id.replace("_", " "),
                    "how": "Manual insert",
                    "signal": f"Returned {len(rows)} rows.",
                    "embed_text": summary_text,
                    "tables_used": [],
                    "key_columns": [],
                    "tags": ["manual", capsule_type],
                    "ttl_hours": 24,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": datetime.now(timezone.utc).isoformat(),
                    "is_stale": False,
                    "staleness_trigger": "manual",
                    "result_rows": rows,
                    "sql": sql_text,
                    "anomaly_score": 0.0,
                    "trend_direction": "flat",
                    "related_capsule_ids": [],
                    "relationship_types": [],
                }
                upsert_capsule(COLLECTION_ANALYTICAL, capsule_id, embed_single(summary_text), payload)
            st.success(f"Inserted {capsule_id}")

with reset_tab:
    st.subheader("Reset")
    st.warning("This clears all Qdrant collections.")
    confirm = st.checkbox("I understand this action cannot be undone")
    if confirm and st.button("Reset Vector DB"):
        with st.spinner("Resetting vector store..."):
            reset_all_collections()
        st.success("Vector store reset complete.")
