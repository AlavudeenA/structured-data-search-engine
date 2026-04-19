"""Streamlit UI entry point for the analytic search engine."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.app_constants import (
    COLLECTION_ANALYTICAL,
    COLLECTION_LINKED,
    COLLECTION_SCHEMA,
    INTENT_ANALYTICAL,
    INTENT_HYBRID,
    INTENT_OPERATIONAL,
    INTENT_STRUCTURED,
    UI_MAX_HISTORY,
)
from src.capsule_builder.relationship_builder import load_graph
from src.capsule_builder.store_manager import generate_all_capsule_collections, refresh_data_only, schema_refresh
from src.capsule_builder.user_capsule_builder import build_single_capsule, enrich_user_capsule_metadata, generate_capsule_sql
from src.capsule_history import available_snapshot_dates, load_capsules_in_range
from src.data_fingerprint import check_schema_and_refresh_if_needed
from src.query_engine.activity_comparator import compare_periods
from src.user_capsules import delete_user_capsule_def, load_user_capsule_defs, save_user_capsule_def
from src.query_engine.orchestrator import handle_query
from src.vector_store import collection_counts, delete_by_capsule_id, reset_all_collections, scroll_all
from src.database_connection import execute_select

st.set_page_config(page_title="Analytical Search Engine", page_icon="", layout="wide")

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
        "schema_checked": False,
        "_uc_preview": None,
        "_uc_intent": "",
        "_uc_sql_input": "",
        "_uc_what_input": "",
        "activity_baseline": {},
        "activity_comparison": {},
        "activity_base_dates": None,
        "activity_comp_dates": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

if not st.session_state.schema_checked:
    check_schema_and_refresh_if_needed()
    st.session_state.schema_checked = True


_ROUTE_STYLE = {
    "sql_direct":               ("steelblue", "SQL Direct"),
    "sql_with_schema_guidance": ("steelblue", "SQL + Schema"),
    "capsule_direct":           ("green",     "Capsule Answer"),
    "vector_retrieval":         ("purple",    "Vector Retrieval"),
    "hybrid":                   ("orange",    "Hybrid"),
    "sql_generation_failed":    ("crimson",   "Generation Failed"),
}


def _pill(color: str, text: str) -> None:
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:999px;font-size:0.9rem;'>{text}</span>",
        unsafe_allow_html=True,
    )


def render_intent_badge(intent: str, confidence: float) -> None:
    color, label = INTENT_BADGE.get(intent, ("gray", intent.title()))
    _pill(color, f"{label} ({confidence:.0%})")


def render_route_badge(route: str) -> None:
    color, label = _ROUTE_STYLE.get(
        route, ("gray", route.replace("_", " ").title())
    )
    _pill(color, label)


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
    _counts = collection_counts()
    st.json(_counts)
    _uc_count = len(load_user_capsule_defs())
    if _uc_count:
        st.caption(f"↳ {_uc_count} user-created of {_counts.get('analytical_capsules', 0)} analytical")

st.title("Analytical Search Engine")

ask_tab, generate_tab, insert_tab, activity_tab, telemetry_tab, explorer_tab, graph_tab, reset_tab = st.tabs(
    [
        "Ask Question",
        "Generate Capsules",
        "Insert Capsule",
        "Data Activity",
        "Telemetry",
        "Capsule Explorer",
        "Capsule Graph",
        "Reset Capsules",
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
            st.markdown("**Route taken**")
            render_route_badge(result.route_taken)

        if result.data_refreshed:
            st.info("Data changed since last query — capsules were automatically refreshed before answering.")

        st.markdown("### Answer")
        st.markdown(result.answer)

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
            f"{actual_counts.get(COLLECTION_LINKED, 0)} linked capsules."
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
        st.success(f"Refreshed {summary.analytical_count} analytical capsules. Schema-context and linked capsules were kept.")
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

    if col3.button("Schema Refresh", type="primary", use_container_width=True):
        progress_rows: list[dict] = []

        def on_progress(capsule_id: str, status: str, preview: str, collection: str = "analytical_capsules") -> None:
            progress_rows.append({"collection": collection, "capsule_id": capsule_id, "status": status, "signal_preview": preview})

        with st.spinner("Refreshing schema and rebuilding all collections..."):
            summary = schema_refresh(progress_callback=on_progress)
        st.success(
            f"Schema refresh complete. Schema changed: {summary.schema_changed}. Rebuilt {summary.analytical_count} analytical, {summary.schema_count} schema_context, and {summary.linked_count} linked capsules."
        )
        if progress_rows:
            st.dataframe(progress_rows, use_container_width=True)

    if col4.button("Generate Capsule Definitions", type="primary", use_container_width=True):
        from src.capsule_builder.definitions_generator import (
            generate_capsule_definitions_via_llm,
            save_generated_definitions,
            validate_definitions_output,
        )

        with st.spinner(
            "Reading live schema → building prompt → calling language model → generating 35+ capsules… "
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
        related = scroll_all(COLLECTION_LINKED)
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
                    "what": (
                        capsule.get("what")
                        or capsule.get("summary")
                        or capsule.get("signal")
                        or "-"
                    ),
                    "how": (
                        capsule.get("how")
                        or (
                            f"Auto-generated {capsule.get('risk_level', '')} risk alert from "
                            f"{capsule.get('entity_type', '')} capsule. "
                            f"Linked from: {', '.join(capsule.get('linked_from') or [])}."
                            if capsule.get("risk_level")
                            else None
                        )
                        or (
                            f"Join path across: {', '.join(capsule.get('tables') or [])}. "
                            f"Key join columns: {', '.join(capsule.get('join_columns') or [])}."
                            if capsule.get("tables")
                            else None
                        )
                        or "-"
                    ),
                    "priority": capsule.get("priority", "-"),
                    "signal_method": capsule.get("signal_method", "-"),
                    "tables_used": ", ".join(capsule.get("tables_used") or capsule.get("tables") or []),
                    "tags": ", ".join(capsule.get("tags", [])),
                    "staleness_trigger": capsule.get("staleness_trigger", "-"),
                    "ttl_hours": capsule.get("ttl_hours", "-"),
                    "anomaly_score": capsule.get("anomaly_score", "-"),
                    "trend_direction": capsule.get("trend_direction", "-"),
                    "linked_count": len(capsule.get("linked_capsule_ids", [])),
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
                "linked_count":      st.column_config.NumberColumn("Linked",            width="small"),
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
            st.write(list(zip(capsule.get("linked_capsule_ids", []), capsule.get("relationship_types", []))))
            st.write("**Source SQL**")
            st.code(capsule.get("sql", capsule.get("sql_template", "")), language="sql")

            delete_collection = COLLECTION_ANALYTICAL
            if "summary" in capsule and "tables" in capsule:
                delete_collection = COLLECTION_SCHEMA
            elif "linked_from" in capsule:
                delete_collection = COLLECTION_LINKED
            if st.button(f"Delete {selected_capsule}"):
                delete_by_capsule_id(delete_collection, selected_capsule)
                delete_user_capsule_def(selected_capsule)  # no-op if not a user capsule
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
        st.dataframe(scroll_all(COLLECTION_LINKED), use_container_width=True)
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
    st.subheader("Create User Capsule")
    st.caption("Describe what you want to measure — AI writes the SQL. Or write SQL directly. AI derives all other metadata.")

    # ── Step 0: Intent → SQL ──────────────────────────────────────────────────
    st.markdown("**Intent** *(optional — describe what to measure in plain English)*")
    intent_val = st.text_area(
        "intent_area",
        label_visibility="collapsed",
        key="_uc_intent",
        height=80,
        placeholder="e.g. Show me brokers with the highest trade rejection rate in the last 90 days",
    )
    gen_btn_disabled = not (intent_val or "").strip()
    if st.button("Generate SQL from Intent", disabled=gen_btn_disabled, key="_gen_sql_btn"):
        with st.spinner("Generating SQL…"):
            generated = generate_capsule_sql(intent_val.strip())
        if generated:
            st.session_state["_uc_sql_input"] = generated
            st.session_state["_uc_what_input"] = intent_val.strip()
            st.rerun()
        else:
            st.error("SQL generation failed — try rephrasing or write SQL manually.")

    st.divider()

    # ── Step 1: User fills only what they know ────────────────────────────────
    with st.form("capsule_form"):
        c_name = st.text_input("Name *", placeholder="e.g. High Risk Broker Trades")
        c_sql = st.text_area(
            "SQL *", height=180,
            key="_uc_sql_input",
            placeholder=(
                "SELECT tr.BrokerDealerID AS broker_id,\n"
                "       COUNT(*) AS total_requests,\n"
                "       SUM(CASE WHEN aw.Decision = 'Rejected' THEN 1 ELSE 0 END) AS rejected\n"
                "FROM TradeRequest tr\n"
                "JOIN ApprovalWorkflow aw ON tr.TradeRequestID = aw.TradeRequestID\n"
                "GROUP BY tr.BrokerDealerID\n"
                "ORDER BY rejected DESC\n"
                "LIMIT 50"
            ),
        )
        c_what = st.text_input(
            "What does this measure? *",
            key="_uc_what_input",
            placeholder="e.g. Rejection rate per broker over the last 90 days",
        )
        c_priority = st.selectbox("Priority", ["P3", "P1", "P2", "P4"])
        submitted = st.form_submit_button("Validate & Enrich", type="primary")

    if submitted:
        if not c_name.strip() or not c_sql.strip() or not c_what.strip():
            st.error("Name, SQL, and What are required.")
        else:
            with st.spinner("Validating SQL and enriching metadata with AI…"):
                try:
                    preview_rows = execute_select(c_sql, max_rows=50)
                except Exception as exc:
                    st.session_state.pop("_uc_preview", None)
                    st.error(f"SQL error: {exc}")
                    preview_rows = None

                if preview_rows is not None:
                    enriched = enrich_user_capsule_metadata(c_what.strip(), c_sql.strip(), preview_rows)
                    st.session_state["_uc_preview"] = {
                        "name":     c_name.strip(),
                        "sql":      c_sql.strip(),
                        "what":     c_what.strip(),
                        "priority": c_priority,
                        "rows":     preview_rows,
                        "enriched": enriched,
                    }

    # ── Step 2: Show preview + enriched metadata, then Save & Build ──────────
    preview = st.session_state.get("_uc_preview")
    if preview:
        st.success(f"SQL valid — {len(preview['rows'])} rows returned.")

        st.markdown("**Preview (up to 50 rows)**")
        st.dataframe(preview["rows"], use_container_width=True)

        enriched = preview["enriched"]
        with st.expander("AI-enriched metadata — review before saving", expanded=True):
            col_l, col_r = st.columns(2)
            col_l.markdown(f"**Type:** `{enriched['capsule_type']}`")
            col_r.markdown(f"**TTL:** `{enriched['ttl_hours']}h` · **Signal:** `{enriched['signal_method']}`")
            st.markdown(f"**How:** {enriched['how']}")
            st.markdown(f"**Tables used:** {', '.join(enriched['tables_used']) or '—'}")
            st.markdown(f"**Key columns:** {', '.join(enriched['key_columns']) or '—'}")
            st.markdown(f"**Tags:** {', '.join(enriched['tags']) or '—'}")
            st.markdown(f"**Staleness trigger:** {enriched['staleness_trigger']}")
            st.markdown(f"**Embed text:** {enriched['embed_text_template']}")

        if st.button("Save & Build Capsule", type="primary"):
            import re as _re
            capsule_id = "user_" + _re.sub(r"[^a-z0-9]+", "_", preview["name"].lower()).strip("_")
            capsule_def = {
                "capsule_id":          capsule_id,
                "capsule_type":        enriched["capsule_type"],
                "priority":            preview["priority"],
                "what":                preview["what"],
                "how":                 enriched["how"],
                "sql":                 preview["sql"],
                "signal_method":       enriched["signal_method"],
                "embed_text_template": enriched["embed_text_template"],
                "ttl_hours":           enriched["ttl_hours"],
                "tags":                enriched["tags"],
                "tables_used":         enriched["tables_used"],
                "key_columns":         enriched["key_columns"],
                "staleness_trigger":   enriched["staleness_trigger"],
                "linked_capsule_ids":  [],
                "relationship_types":  [],
            }
            with st.spinner("Building and linking capsule…"):
                save_user_capsule_def(capsule_def)
                ok = build_single_capsule(capsule_def)

            if ok:
                st.success(f"Capsule `{capsule_id}` saved, built, and linked to related capsules.")
                st.session_state.pop("_uc_preview", None)
                st.rerun()
            else:
                st.error("Build failed — definition saved but not indexed. Check logs.")

    # ── My Capsules (list + delete) ───────────────────────────────────────────
    st.divider()
    st.subheader("My Capsules")
    user_caps = load_user_capsule_defs()
    if not user_caps:
        st.info("No user capsules created yet.")
    else:
        for uc in user_caps:
            col_info, col_del = st.columns([5, 1])
            col_info.markdown(f"**{uc['capsule_id']}** — {uc['what']} `{uc['capsule_type']}` `{uc['priority']}`")
            if col_del.button("Delete", key=f"del_{uc['capsule_id']}"):
                delete_user_capsule_def(uc["capsule_id"])
                delete_by_capsule_id(COLLECTION_ANALYTICAL, uc["capsule_id"])
                st.success(f"Deleted {uc['capsule_id']}")
                st.rerun()

with activity_tab:
    st.subheader("Data Activity")
    st.caption(
        "Compare analytical capsule snapshots across two time periods. "
        "Snapshots are saved automatically when data changes are detected during a query."
    )

    snapshot_dates = available_snapshot_dates()

    if not snapshot_dates:
        st.info(
            "No capsule snapshots yet. Snapshots are created automatically the first time "
            "a data change is detected while running a query."
        )
    else:
        earliest = snapshot_dates[0]
        latest = snapshot_dates[-1]

        st.markdown("**Available snapshot dates:** " + ", ".join(d.strftime("%m/%d/%Y") for d in snapshot_dates))
        st.divider()

        col_base, col_comp = st.columns(2)

        with col_base:
            st.markdown("### Baseline Period")
            base_from = st.date_input(
                "From", value=earliest, min_value=earliest, max_value=latest, key="base_from"
            )
            base_to = st.date_input(
                "To", value=latest, min_value=earliest, max_value=latest, key="base_to"
            )

        with col_comp:
            st.markdown("### Comparison Period")
            comp_from = st.date_input(
                "From", value=earliest, min_value=earliest, max_value=latest, key="comp_from"
            )
            comp_to = st.date_input(
                "To", value=latest, min_value=earliest, max_value=latest, key="comp_to"
            )

        st.divider()

        if st.button("Load Capsules", type="secondary"):
            if base_from > base_to:
                st.error("Baseline Period: 'From' date must be on or before 'To' date.")
            elif comp_from > comp_to:
                st.error("Comparison Period: 'From' date must be on or before 'To' date.")
            else:
                st.session_state.activity_baseline = load_capsules_in_range(base_from, base_to)
                st.session_state.activity_comparison = load_capsules_in_range(comp_from, comp_to)
                st.session_state.activity_base_dates = (base_from, base_to)
                st.session_state.activity_comp_dates = (comp_from, comp_to)
                st.session_state.activity_base_selected = set(st.session_state.activity_baseline.keys())
                st.session_state.activity_comp_selected = set(st.session_state.activity_comparison.keys())

        baseline_caps: dict = st.session_state.get("activity_baseline", {})
        comparison_caps: dict = st.session_state.get("activity_comparison", {})

        if baseline_caps or comparison_caps:
            col_bl, col_cl = st.columns(2)

            with col_bl:
                st.markdown(f"**Baseline — {len(baseline_caps)} unique capsule(s)**")
                if baseline_caps:
                    all_base = st.checkbox("Select all (Baseline)", value=True, key="base_all")
                    base_selected: set[str] = set()
                    for cid, cap in baseline_caps.items():
                        default = all_base
                        checked = st.checkbox(
                            f"{cid} · {cap.get('capsule_type','?')} · anomaly={cap.get('anomaly_score','?')}",
                            value=default,
                            key=f"base_cb_{cid}",
                        )
                        if checked:
                            base_selected.add(cid)
                else:
                    st.info("No snapshots found in Baseline range.")
                    base_selected = set()

            with col_cl:
                st.markdown(f"**Comparison — {len(comparison_caps)} unique capsule(s)**")
                if comparison_caps:
                    all_comp = st.checkbox("Select all (Comparison)", value=True, key="comp_all")
                    comp_selected: set[str] = set()
                    for cid, cap in comparison_caps.items():
                        default = all_comp
                        checked = st.checkbox(
                            f"{cid} · {cap.get('capsule_type','?')} · anomaly={cap.get('anomaly_score','?')}",
                            value=default,
                            key=f"comp_cb_{cid}",
                        )
                        if checked:
                            comp_selected.add(cid)
                else:
                    st.info("No snapshots found in Comparison range.")
                    comp_selected = set()

            st.divider()

            if st.button("Compare Periods", type="primary"):
                final_baseline = {k: v for k, v in baseline_caps.items() if k in base_selected}
                final_comparison = {k: v for k, v in comparison_caps.items() if k in comp_selected}

                if not final_baseline and not final_comparison:
                    st.warning("Select at least one capsule on either side to compare.")
                else:
                    base_dates = st.session_state.get("activity_base_dates", (base_from, base_to))
                    comp_dates = st.session_state.get("activity_comp_dates", (comp_from, comp_to))
                    with st.spinner("Comparing periods via LLM..."):
                        result = compare_periods(
                            baseline_capsules=final_baseline,
                            comparison_capsules=final_comparison,
                            baseline_start=base_dates[0],
                            baseline_end=base_dates[1],
                            comparison_start=comp_dates[0],
                            comparison_end=comp_dates[1],
                        )
                    st.markdown("### Analysis")
                    st.markdown(result)

with reset_tab:
    st.subheader("Reset Capsules")
    st.warning("This removes all indexed capsules (analytical, schema, linked). All capsule definitions are preserved — run **Generate All Capsules** in the **Generate Capsules** tab to rebuild.")
    user_caps_count = len(load_user_capsule_defs())
    if user_caps_count:
        st.info(
            f"You have {user_caps_count} user-created capsule(s). Their definitions are stored in "
            "`data/user_capsules.json` and will be preserved — but their vectors will be cleared. "
            "Run **Generate All Capsules** or **Refresh Data** after reset to rebuild them."
        )
    confirm = st.checkbox("I understand this action cannot be undone")
    if confirm and st.button("Reset Vector DB"):
        with st.spinner("Resetting vector store..."):
            st.session_state.explorer_capsules = []
            reset_all_collections()
            with sidebar_collections.container():
                st.markdown("### Collections")
                st.json(collection_counts())
        st.success("Vector store reset complete. User capsule definitions preserved — rebuild to reindex them.")
