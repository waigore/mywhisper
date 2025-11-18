# Examples

The legacy scripts in this directory (`transcribe_audio.py`, `diarize_audio.py`, etc.) demonstrate ad-hoc notebooks and should now be considered reference material only.

Prefer the new high-level APIs exposed by the `mywhisper` package:

- `mywhisper.transcribe.PodcastTranscriber` for Whisper transcription.
- `mywhisper.diarize.DiarizationPipeline` for PyAnnote-based diarization.
- `mywhisper.assign.TranscriptAssigner` for speaker name inference.
- `mywhisper.prettify.TranscriptPrettifier` for readable transcript generation and condensed JSON.
- `mywhisper.thematize.EpisodeThematizer` for LLM-driven per-segment thematic summaries.
- `mywhisper.podcasts.PodcastCatalog` and `ApplePodcastsImporter` for catalog management.

Use `prettify_and_thematize.py` to run the final formatting steps. Prettify runs after diarization to produce a readable transcript and `{episode_key}_condensed.json`. Thematize consumes the condensed JSON and writes `{episode_key}_with_themes.json`. If you want real speaker names in the readable transcript, run the assign step after thematize.

See the project `README.md` for quick-start snippets that tie these components together.

