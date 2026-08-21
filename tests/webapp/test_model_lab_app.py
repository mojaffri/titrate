from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_model_lab_renders_metrics_and_interactive_predictions() -> None:
    page = Path(__file__).resolve().parents[2] / "webapp" / "pages" / "1_Model_Lab.py"
    app = AppTest.from_file(str(page), default_timeout=120).run()

    assert not app.exception
    assert any("Model evidence lab" in markdown.value for markdown in app.markdown)
    assert len(app.dataframe) >= 2
    assert len(app.slider) >= 5
