from __future__ import annotations

from pathlib import Path
import tomllib

from streamlit.testing.v1 import AppTest

from streamlit_app import _hero_dashboard_preview


def test_homepage_hero_uses_native_streamlit_rendering_and_keeps_review_flow():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    visible_text = " ".join(
        str(element.value)
        for collection in (app.markdown, app.caption)
        for element in collection
    )
    assert "Discover :green[career opportunities]" in visible_text
    assert "AI-assisted career intelligence" in visible_text
    assert "Evidence-led matching" in visible_text
    assert "Build a profile" in visible_text
    assert "Explore roles" in visible_text
    assert "Review evidence" in visible_text
    assert "Plan next steps" in visible_text
    assert "Career direction" in visible_text
    assert "Experience and background" in visible_text
    assert "Work preferences" in visible_text
    labels = [element.label for element in app.text_area]
    assert "Desired job titles (one per line) *" in labels
    assert "Relevant skills (one per line) *" in labels
    assert "Responsibilities or project description (one item per line)" in labels
    assert not any("?" in label for label in labels)
    assert not any("<div" in str(element.value) for element in app.markdown)
    assert "Review candidate profile" in [button.label for button in app.button]
    assert "Run Career Intelligence Agent" in [button.label for button in app.button]

    app.text_area(key="desired_titles").set_value("AI Consultant")
    app.text_area(key="skills").set_value("AI implementation")
    app.text_input(key="experience_title_0").set_value("AI project consultant")
    app.text_area(key="experience_responsibilities_0").set_value(
        "Implemented an AI use case with stakeholders."
    )
    app.run(timeout=15)
    next(button for button in app.button if button.label == "Review candidate profile").click().run(timeout=15)
    confirmation = next(
        checkbox for checkbox in app.checkbox
        if checkbox.label.startswith("I reviewed this profile")
    )
    confirmation.set_value(True).run(timeout=15)
    run_button = next(button for button in app.button if button.label == "Run Career Intelligence Agent")
    assert not run_button.disabled
    assert app.session_state.get("run_result") is None
    assert not app.exception


def test_mode_notices_are_informational_and_profile_validation_remains_a_warning():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    assert not app.exception
    demo_notice = " ".join(str(element.value) for element in app.info)
    assert "Demo Mode uses clearly labeled synthetic job examples" in demo_notice
    assert any("Add at least one desired job title." in element.value for element in app.warning)

    app.radio(key="mode").set_value("Live mode").run(timeout=15)
    assert not app.exception
    live_notices = " ".join(str(element.value) for element in app.info)
    assert "Live mode contacts external services" in live_notices
    assert "Live Mode searches real job providers" in live_notices
    assert any("Add at least one desired job title." in element.value for element in app.warning)
    assert not any("may incur charges" in element.value for element in app.warning)


def test_alert_css_uses_dark_surfaces_and_a_scoped_validation_accent():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "background: #112D3A !important" in source
    assert "border: 1px solid #286073 !important" in source
    assert ".st-key-cia_validation_notice [data-testid=\"stAlert\"]" in source
    assert "border-left: 3px solid var(--cia-amber) !important" in source


def test_experience_form_has_separate_optional_dates_and_total_experience_label():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)

    app.number_input(key="experience_count").set_value(2).run(timeout=15)
    assert not app.exception
    assert app.text_input(key="experience_title_0").label == "Job or project title"
    assert app.text_input(key="experience_start_date_0").label == "Start date (optional)"
    assert app.text_input(key="experience_end_date_0").label == "End date (optional)"
    assert app.checkbox(key="experience_ongoing_0").label == "Ongoing role/project (Present)"
    assert app.text_input(key="experience_start_date_1")
    assert app.text_input(key="experience_end_date_1")

    app.checkbox(key="years_enabled").set_value(True).run(timeout=15)
    assert any(item.label == "Total professional experience (years)" for item in app.number_input)
    assert not app.exception

    app.radio(key="mode").set_value("Live mode").run(timeout=15)
    assert not app.exception
    assert app.text_input(key="experience_title_0").label == "Job or project title"
    assert app.text_input(key="experience_start_date_1").label == "Start date (optional)"


def test_recommendation_and_review_cards_render_job_posting_details_in_demo_and_live_modes():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    for mode in ("Demo mode", "Live mode"):
        app = AppTest.from_file(str(app_path)).run(timeout=15)
        app.radio(key="mode").set_value(mode).run(timeout=15)
        app.text_area(key="desired_titles").set_value("AI Transformation Consultant")
        app.text_area(key="skills").set_value("AI implementation")
        app.text_input(key="experience_title_0").set_value("AI Consultant")
        app.text_area(key="experience_responsibilities_0").set_value("Implemented AI workflows with project teams.")
        app.run(timeout=15)
        next(button for button in app.button if button.label == "Review candidate profile").click().run(timeout=15)

        recommendation = {
            "job": {"title": "AI Transformation Consultant", "company": "Example Co", "location": "Munich or Cologne",
                "work_model": "remote", "source": "fixture", "source_url": "https://example.org/jobs/1",
                "description": "<h3>Location</h3><p>Based in Munich or Cologne, with onboarding in Zurich for one month.</p>"
                    "<h3>Key Responsibilities</h3><p>Lead AI implementations and transformation delivery.</p>"},
            "score": 80, "evidence_coverage": 75, "strengths": ["Project delivery evidence."], "gaps": [],
            "requirements_to_verify": [], "career_direction": {"status": "aligned", "evidence": ["Matches target role."]},
        }
        review = {
            **recommendation, "job": {**recommendation["job"], "title": "Project Manager"},
            "career_direction": {"status": "uncertain", "evidence": ["Duties need review."]},
            "reasons": ["Career-direction alignment is uncertain."], "support_status": "provisional",
        }
        payload = {"workflow_status": "completed", "report_data": {
            "search_scope": {"offline": mode == "Demo mode", "provider_diagnostics": {}},
            "workflow": {"status": "completed"}, "discovered_jobs": 2, "evaluated_jobs": 2,
            "recommendations": [recommendation], "potential_matches_requiring_review": [review],
            "market_intelligence": {"status": "skipped_offline", "evidence": []},
            "intelligence_synthesis": {"status": "skipped_offline"}, "personalized_career_intelligence": {},
            "esco_enrichment": {"status": "skipped_offline"}, "company_research": {"status": "skipped_offline"},
        }}
        signature = app.session_state["reviewed_signature"]
        app.session_state["run_result"] = payload
        app.session_state["run_signature"] = signature
        app.run(timeout=15)

        assert not app.exception, (mode, app.exception)
        rendered = " ".join(str(element.value) for collection in (app.markdown, app.caption)
                            for element in collection)
        assert "Key job requirements & working conditions" in rendered
        assert "Provider-listed work model: Remote" in rendered
        assert "onboarding in Zurich" in rendered
        assert "Project Manager" in rendered
        download_buttons = app.get("download_button")
        assert {button.label for button in download_buttons} == {
            "Download Markdown report", "Download JSON report", "Download PDF Report",
        }


def test_streamlit_app_imports_the_experience_title_helper():
    # Importing streamlit_app establishes the src-layout path just as the app entry point does.
    from career_agent.streamlit_support import split_experience_title

    assert split_experience_title("AI Consultant (Jan 2024 - Present)") == (
        "AI Consultant", "Jan 2024", "", True,
    )


def test_streamlit_theme_config_matches_the_application_palette():
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    theme = config["theme"]
    assert theme["base"] == "dark"
    assert theme["primaryColor"].casefold() == "#1eea91"
    assert theme["backgroundColor"].casefold() == "#07151c"
    assert theme["secondaryBackgroundColor"].casefold() == "#10222d"
    assert theme["textColor"].casefold() == "#f2faf8"


def test_hero_dashboard_preview_is_fixed_illustration_without_fabricated_metrics():
    preview = _hero_dashboard_preview()
    assert "ILLUSTRATIVE PREVIEW" in preview
    for capability in (
        "Candidate context", "Opportunity discovery", "Transparent evaluation",
        "Traceable job evidence", "Cited market insights",
    ):
        assert capability in preview
    assert "score" not in preview.casefold()
    assert "success rate" not in preview.casefold()
