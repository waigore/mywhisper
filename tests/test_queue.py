from __future__ import annotations

import time

from mywhisper.myw.services.queue import QueueController


def test_queue_enqueue_and_current_status():
    queue = QueueController(enqueue_delay=0.0)
    queue.enqueue("ep1")
    assert queue.has_pending()

    item = queue.next_item()
    assert item is not None
    assert item.episode_id == "ep1"
    assert queue.current_episode_id() == "ep1"

    status = queue.current_status()
    assert status.active
    assert status.episode_id == "ep1"

    queue.stop_current()
    assert queue.should_stop()
    queue.set_status("ep1", "Stopped", "Paused")
    assert queue.get_status("ep1") == ("Stopped", "Paused")

    queue.release_current()
    assert queue.current_episode_id() is None
    assert not queue.should_stop()
    assert queue.current_status().active is False


def test_queue_resume_priority_and_snapshot():
    queue = QueueController(enqueue_delay=0.05)
    queue.enqueue("ep1")
    queue.enqueue("ep2")
    queue.resume_episode("ep3")

    snapshot = queue.snapshot_status()
    assert "ep3" in snapshot
    assert snapshot["ep3"][0] == "In progress"

    first = queue.next_item()
    assert first.episode_id == "ep3"
    queue.release_current()

    time.sleep(0.06)
    second = queue.next_item()
    assert second.episode_id == "ep1"
    queue.release_current()

    third = queue.next_item()
    assert third.episode_id == "ep2"
    queue.release_current()


def test_queue_shutdown_returns_none():
    queue = QueueController()
    queue.request_shutdown()
    assert queue.next_item() is None

