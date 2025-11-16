# Examples

The legacy scripts in this directory (`transcribe_audio.py`, `diarize_audio.py`, etc.) demonstrate ad-hoc notebooks and should now be considered reference material only.

Prefer the new high-level APIs exposed by the `mywhisper` package:

- `mywhisper.transcribe.PodcastTranscriber` for Whisper transcription.
- `mywhisper.diarize.DiarizationPipeline` for PyAnnote-based diarization.
- `mywhisper.assign.TranscriptAssigner` for speaker name inference.
- `mywhisper.prettify.TranscriptPrettifier` for readable transcript generation.
- `mywhisper.thematize.EpisodeThematizer` for LLM-driven thematic summaries.
- `mywhisper.podcasts.PodcastCatalog` and `ApplePodcastsImporter` for catalog management.

Use `prettify_and_thematize.py` to run only the final formatting steps once speaker assignments are available.

See the project `README.md` for quick-start snippets that tie these components together.

