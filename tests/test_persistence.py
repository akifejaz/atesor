import os
import tempfile
import unittest

from src.persistence import SessionStore
from src.state import create_initial_state, BuildStatus


class TestSessionStore(unittest.TestCase):
    def test_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            store = SessionStore(db_path=db_path)

            session_id = store.create_session("https://github.com/example/repo", 3)
            state = create_initial_state("https://github.com/example/repo", max_attempts=3)
            state.build_status = BuildStatus.PENDING
            state.log_event("decision", {"action": "SCOUT"})

            store.save_snapshot(session_id, 1, "planner", state)
            store.save_events(session_id, 1, state.audit_trail)
            store.finish_session(session_id, state, exit_code=0)

            latest = store.get_latest_snapshot(session_id)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["step"], 1)
            self.assertEqual(latest["node_name"], "planner")
            self.assertEqual(latest["state"]["repo_name"], "repo")


if __name__ == "__main__":
    unittest.main()
