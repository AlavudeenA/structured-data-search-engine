"""Streamlit UI entry point for the analytic search engine."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.app_constants import (
    COLLECTION_ANALYTICAL,
    COLLECTION_RELATED,
    COLLECTION_SCHEMA,
    INTENT_ANALYTICAL,
    INTENT_HYBRID,
    INTENT_OPERATIONAL,
    INTENT_STRUCTURED,
    UI_MAX_HISTORY,
)
from src.capsule_builder.relationship_builder import load_graph
from src.capsule_builder.store_manager import collection_stats, generate_all_capsule_collections, refresh_data_only, schema_refresh
from src.query_engine.orchestrator import handle_query
from src.vector_store import clear_collection, collection_counts, delete_by_capsule_id, reset_all_collections, scroll_all, upsert_capsule
from src.database_connection import execute_select
from src.embedding import embed_single

st.set_page_config(page_title="Analytical Search Engine", page_icon="🔍", layout="wide")

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


sidebar_collections = st.sidebar.empty()
with sidebar_collections.container():
    st.markdown("### Collections")
    st.json(collection_counts())

st.title("Analytical Search Engine")

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
    question = st.text_area("Question", height=120, placeholder="Ask a question about your database...")
    history = [item["question"] for item in st.session_state.question_history[-UI_MAX_HISTORY:]]
    if history:
        selected = st.selectbox("Recent Questions", [""] + history)
        if selected and not question:
            question = selected

    if st.button("Run Question", type="primary") and question.strip():
        st.session_state.last_question = question.strip()
        with st.spinner("Running analytical pipeline..."):
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
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="small")

    if col1.button("Generate All Capsules", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting full build")
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str, collection: str = "analytical_capsules") -> None:
            progress_rows.append({"collection": collection, "capsule_id": capsule_id, "status": status, "signal_preview": preview})
            total = max(len(progress_rows), 1)
            progress.progress(min(total / 32, 1.0), text=f"Built {capsule_id}")

        with st.spinner("Building all collections..."):
            summary = generate_all_capsule_collections(progress_callback=on_progress)
        progress.progress(1.0, text="Completed")
        actual_counts = collection_counts()  # single source of truth
        st.success(
            f"Generated {actual_counts.get(COLLECTION_ANALYTICAL, 0)} analytical, "
            f"{actual_counts.get(COLLECTION_SCHEMA, 0)} schema_context, and "
            f"{actual_counts.get(COLLECTION_RELATED, 0)} related capsules."
        )
        st.write(f"Relationship graph edges: {summary.graph_edge_count}")
        st.dataframe(progress_rows, use_container_width=True)

        # Sidebar updated from same actual_counts — guaranteed to match success message
        with sidebar_collections.container():
            st.markdown("### Collections")
            st.json(actual_counts)

    if col2.button("Refresh Data", type="primary", use_container_width=True):
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str, collection: str = "analytical_capsules") -> None:
            progress_rows.append({"collection": collection, "capsule_id": capsule_id, "status": status, "signal_preview": preview})

        with st.spinner("Refreshing analytical capsules from saved plan..."):
            summary = refresh_data_only(progress_callback=on_progress)
        st.success(f"Refreshed {summary.analytical_count} analytical capsules. Schema-context and related capsules were kept.")
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

    if col3.button("Schema Refresh", type="primary", use_container_width=True):
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str, collection: str = "analytical_capsules") -> None:
            progress_rows.append({"collection": collection, "capsule_id": capsule_id, "status": status, "signal_preview": preview})

        with st.spinner("Refreshing schema and rebuilding all collections..."):
            summary = schema_refresh(progress_callback=on_progress)
        st.success(
            f"Schema refresh complete. Schema changed: {summary.schema_changed}. Rebuilt {summary.analytical_count} analytical, {summary.schema_count} schema_context, and {summary.related_count} related capsules."
        )
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

    if col4.button("AI Rebuild Definitions", type="primary", use_container_width=True):
        from src.capsule_builder.definitions_generator import (
            generate_capsule_definitions_via_llm,
            save_generated_definitions,
            validate_definitions_output,
        )

        with st.spinner(
            "Reading live schema → building prompt → calling Groq 70b → generating 35+ capsules… "
            "This takes 30–90 seconds."
        ):
            try:
                raw_output = generate_capsule_definitions_via_llm()
                st.session_state["regen_raw_output"] = raw_output
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                raw_output = None

        if raw_output:
            is_valid, validation_error = validate_definitions_output(raw_output)

            if is_valid:
                # Count capsules by scanning for { "capsule_id" patterns
                import re as _re
                capsule_count = len(_re.findall(r'"capsule_id"\s*:', raw_output))
                st.success(
                    f"✅ LLM generated **{capsule_count} capsules** — valid Python syntax confirmed."
                )
            else:
                st.warning(f"⚠️ Syntax warning: {validation_error}. Review before saving.")

            st.info(
                "Review the generated definitions below. "
                "Click **💾 Save & Activate** to overwrite `capsule_definitions.py`, "
                "then click **Generate All Capsules** to rebuild the vector store."
            )

            with st.expander("Generated CAPSULE_DEFINITIONS (Python)", expanded=True):
                st.code(raw_output, language="python")

    # Save button lives outside the generate block so it persists across reruns
    if st.session_state.get("regen_raw_output"):
        raw_output = st.session_state["regen_raw_output"]
        col_save, col_clear = st.columns([2, 1])
        if col_save.button("💾 Save & Activate", type="primary", key="save_regen"):
            from src.capsule_builder.definitions_generator import save_generated_definitions
            try:
                save_generated_definitions(raw_output)
                st.success(
                    "✅ Saved to `src/business_schema/capsule_definitions.py`. "
                    "Click **Generate All Capsules** to rebuild the vector store."
                )
                st.session_state.pop("regen_raw_output", None)
            except Exception as exc:
                st.error(f"Save failed: {exc}")
        if col_clear.button("🗑️ Discard", key="clear_regen"):
            st.session_state.pop("regen_raw_output", None)
            st.rerun()

with explorer_tab:
    st.subheader("Capsule Explorer")
    if st.button("Load All Capsules"):
        analytical = scroll_all(COLLECTION_ANALYTICAL)
        schema = scroll_all(COLLECTION_SCHEMA)
        related = scroll_all(COLLECTION_RELATED)
        st.session_state.explorer_capsules = analytical + schema + related

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
                    "what": capsule.get("what") or capsule.get("summary", "-"),
                    "how": capsule.get("how", "-"),
                    "priority": capsule.get("priority", "-"),
                    "signal_method": capsule.get("signal_method", "-"),
                    "tables_used": ", ".join(capsule.get("tables_used") or capsule.get("tables") or []),
                    "tags": ", ".join(capsule.get("tags", [])),
                    "staleness_trigger": capsule.get("staleness_trigger", "-"),
                    "ttl_hours": capsule.get("ttl_hours", "-"),
                    "anomaly_score": capsule.get("anomaly_score", "-"),
                    "trend_direction": capsule.get("trend_direction", "-"),
                    "related_count": len(capsule.get("related_capsule_ids", [])),
                    "expires_at": capsule.get("expires_at", "-"),
                }
                for capsule in filtered
            ],
            column_config={
                "capsule_id":        st.column_config.TextColumn("Capsule ID",         width="medium"),
                "type":              st.column_config.TextColumn("Type",               width="small"),
                "what":              st.column_config.TextColumn("What",               width="medium",  help="What this capsule measures — hover a cell to read in full"),
                "how":               st.column_config.TextColumn("How",                width="medium",  help="How the metric is calculated — hover a cell to read in full"),
                "priority":          st.column_config.TextColumn("Priority",           width="small"),
                "signal_method":     st.column_config.TextColumn("Signal Method",      width="small"),
                "tables_used":       st.column_config.TextColumn("Tables",             width="small",   help="Database tables used by this capsule"),
                "tags":              st.column_config.TextColumn("Tags",               width="small",   help="All tags — hover a cell to read in full"),
                "staleness_trigger": st.column_config.TextColumn("Staleness Trigger",  width="small",   help="Event that invalidates this capsule — hover to read in full"),
                "ttl_hours":         st.column_config.NumberColumn("TTL (hrs)",         width="small"),
                "anomaly_score":     st.column_config.NumberColumn("Anomaly",           width="small",   format="%.2f"),
                "trend_direction":   st.column_config.TextColumn("Trend",              width="small"),
                "related_count":     st.column_config.NumberColumn("Related",           width="small"),
                "expires_at":        st.column_config.TextColumn("Expires At",         width="medium"),
            },
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
            elif "related_from" in capsule:
                delete_collection = COLLECTION_RELATED
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
        st.write("### Related Capsules")
        st.dataframe(scroll_all(COLLECTION_RELATED), use_container_width=True)
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
    capsule_id = st.text_input("Capsule ID*", help="Unique identifier in snake_case (e.g. high_risk_trades)")
    
    type_options = [
        "aggregation", "anomaly", "concentration", "correlation", 
        "distribution", "escalation", "forecast", "outlier", 
        "pattern", "profiling", "risk_score", "summary", 
        "threshold_breach", "trend", "Custom..."
    ]
    selected_type = st.selectbox("Capsule Type*", type_options, index=0)
    if selected_type == "Custom...":
        capsule_type = st.text_input("Enter Custom Capsule Type", value="custom_type")
    else:
        capsule_type = selected_type
        
    max_return_rows = st.number_input("Max Rows to Return", min_value=1, max_value=1000, value=50, step=10)
    sql_text = st.text_area("SQL*", height=180)
    summary_text = st.text_area("Summary Text / Embed Text*", height=120)

    with st.expander("Advanced Capsule Properties (Leave blank for LLM auto-generation)"):
        c_priority = st.selectbox("Priority", ["", "P1", "P2", "P3", "P4"], help="Risk severity")
        c_what = st.text_input("What", help="Short description of the capsule")
        c_how = st.text_input("How", help="1-sentence explanation of calculation")
        c_signal_method = st.text_input("Signal Method", help="e.g. threshold_breach, latest_value")
        c_ttl_hours = st.text_input("TTL Hours", help="Enter a number (e.g., 24)")
        c_staleness = st.selectbox("Staleness Trigger", ["", "hourly", "daily", "weekly", "monthly", "manual"])
        c_tags = st.text_input("Tags", help="Comma-separated semantic tags")
        c_tables = st.text_input("Tables Used", help="Comma-separated table names")
        c_keys = st.text_input("Key Columns", help="Comma-separated key columns")

    if st.button("Insert Capsule", type="primary"):
        if not capsule_id or not sql_text or not summary_text:
            st.error("Capsule ID, Type, SQL, and Summary are required.")
        else:
            invalid_sql = False
            rows = []
            
            with st.spinner("Validating SQL query..."):
                try:
                    rows = execute_select(sql_text, max_rows=max_return_rows)
                except Exception as e:
                    invalid_sql = True
                    st.error(f"❌ Invalid SQL Query: {str(e)}")
            
            if not invalid_sql:
                with st.spinner("Determining metadata via LLM and saving..."):
                    from src.llm_service import call_llm_json
                    from src.capsule_builder.append_capsules import append_to_capsule_definitions_file
                    
                    sys_prompt = (
                        "You are a smart data analytics engine. Given a capsule summary and SQL, return a JSON object containing "
                        "ONLY the fields that are strictly missing from the user's manual inputs to complete the definition:\n"
                        "- priority (string): P1, P2, P3, or P4.\n"
                        "- what (string): Short descriptive title.\n"
                        "- how (string): 1 sentence explaining the calculation.\n"
                        "- signal_method (string): e.g., 'max_value', 'threshold_breach', 'count_distinct'.\n"
                        "- ttl_hours (int): Number of hours.\n"
                        "- staleness_trigger (string): 'hourly', 'daily', 'weekly', etc.\n"
                        "- tables_used (list of strings).\n"
                        "- key_columns (list of strings).\n"
                        "- tags (list of strings).\n"
                        "Output ONLY valid JSON."
                    )
                    
                    req_fields = {}
                    if not c_priority: req_fields['priority'] = "?"
                    if not c_what: req_fields['what'] = "?"
                    if not c_how: req_fields['how'] = "?"
                    if not c_signal_method: req_fields['signal_method'] = "?"
                    if not c_ttl_hours: req_fields['ttl_hours'] = "?"
                    if not c_staleness: req_fields['staleness_trigger'] = "?"
                    if not c_tags: req_fields['tags'] = "?"
                    if not c_tables: req_fields['tables_used'] = "?"
                    if not c_keys: req_fields['key_columns'] = "?"
                    
                    llm_meta = {}
                    if req_fields:
                        usr_prompt = f"Capsule ID: {capsule_id}\nType: {capsule_type}\nSummary: {summary_text}\nSQL: {sql_text}\n\nPlease generate these missing fields: {list(req_fields.keys())}"
                        llm_meta = call_llm_json(sys_prompt, usr_prompt, model_slot="groq_intent_model") or {}
                    
                    def merge_str(user_val, llm_val, default):
                        return user_val.strip() if user_val else str(llm_meta.get(llm_val, default))
                        
                    def merge_list(user_val, llm_val):
                        if user_val:
                            return [x.strip() for x in user_val.split(",") if x.strip()]
                        return llm_meta.get(llm_val, [])
                        
                    def merge_int(user_val, llm_val, default):
                        if user_val and user_val.isdigit():
                            return int(user_val)
                        return int(llm_meta.get(llm_val, default))

                    final_priority = merge_str(c_priority, "priority", "P3")
                    final_what = merge_str(c_what, "what", capsule_id.replace("_", " "))
                    final_how = merge_str(c_how, "how", "Manual insert")
                    final_sig_method = merge_str(c_signal_method, "signal_method", "manual")
                    final_ttl = merge_int(c_ttl_hours, "ttl_hours", 24)
                    final_stale = merge_str(c_staleness, "staleness_trigger", "manual")
                    final_tags = ["manual", capsule_type] + merge_list(c_tags, "tags")
                    final_tables = merge_list(c_tables, "tables_used")
                    final_keys = merge_list(c_keys, "key_columns")

                    # Construct exact definition equivalent
                    capsule_def = {
                        "capsule_id": capsule_id,
                        "capsule_type": capsule_type,
                        "priority": final_priority,
                        "what": final_what,
                        "how": final_how,
                        "sql": sql_text.strip(),
                        "signal_method": final_sig_method,
                        "embed_text_template": summary_text,
                        "ttl_hours": final_ttl,
                        "tags": final_tags,
                        "tables_used": final_tables,
                        "key_columns": final_keys,
                        "staleness_trigger": final_stale,
                        "related_capsule_ids": [],
                        "relationship_types": []
                    }
                    
                    # Augment with runtime execution payload
                    payload = dict(capsule_def)
                    payload["signal"] = f"Returned {len(rows)} rows."
                    payload["embed_text"] = summary_text
                    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
                    payload["expires_at"] = datetime.now(timezone.utc).isoformat()
                    payload["is_stale"] = False
                    payload["result_rows"] = rows
                    payload["anomaly_score"] = 0.0
                    payload["trend_direction"] = "flat"
                    
                    # Store vector internally
                    upsert_capsule(COLLECTION_ANALYTICAL, capsule_id, embed_single(summary_text), payload)
                    
                    # Append strictly to source code definitions file
                    file_saved = append_to_capsule_definitions_file(capsule_def)
                    
                st.success(f"✅ Query successful: Retrieved {len(rows)} rows.")
                if file_saved:
                    st.success(f"✅ Permanent Save: Successfully appended '{capsule_id}' directly to capsule_definitions.py!")
                else:
                    st.warning(f"⚠️ Inserted locally, but failed to write to capsule_definitions.py. Check app console.")

with reset_tab:
    st.subheader("Reset")
    st.warning("This clears all Qdrant collections.")
    confirm = st.checkbox("I understand this action cannot be undone")
    if confirm and st.button("Reset Vector DB"):
        with st.spinner("Resetting vector store..."):
            st.session_state.explorer_capsules = []
            reset_all_collections()
            with sidebar_collections.container():
                st.markdown("### Collections")
                st.json(collection_counts())
        st.success("Vector store reset complete.")
