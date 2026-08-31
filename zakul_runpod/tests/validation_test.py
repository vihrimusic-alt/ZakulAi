"""Regression tests for durations, types and explicit feature boundaries."""

import unittest

from zakul_runpod.validation import parse_generate, validate_job


def request(**changes):
    """Build a valid instrumental smoke input."""
    return {"prompt": "Sopilka and bass", "instrumental": True, "duration_seconds": 20, **changes}


class ValidationTests(unittest.TestCase):
    """Unknown, malformed or excessive requests must fail before GPU work."""

    def test_instrumental_success(self):
        parsed = parse_generate(request())
        self.assertEqual(parsed.lyrics, "[Instrumental]")
        self.assertEqual(parsed.duration, 20)

    def test_lyrics_remain_in_original_language(self):
        raw = "[Verse]\nТи залишся поруч зі мною"
        parsed = parse_generate(request(instrumental=False, lyrics=raw))
        self.assertEqual(parsed.lyrics, raw)
        self.assertEqual(parsed.language, "unknown")

    def test_explicit_short_clip_not_replaced_by_song_default(self):
        self.assertEqual(parse_generate(request(duration_seconds=2.12)).duration, 2.12)

    def test_duration_required(self):
        with self.assertRaises(ValueError):
            parse_generate({"prompt": "Bass", "instrumental": True})

    def test_invalid_durations(self):
        for duration in [None, True, "20", 0, -1, 241, float("nan"), float("inf")]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                parse_generate(request(duration_seconds=duration))

    def test_string_false_is_not_true(self):
        with self.assertRaises(ValueError):
            parse_generate(request(instrumental="false"))

    def test_no_lyrics_cannot_masquerade_as_vocal_generation(self):
        with self.assertRaises(ValueError):
            parse_generate(request(instrumental=False))

    def test_instrumental_does_not_discard_lyrics(self):
        with self.assertRaises(ValueError):
            parse_generate(request(lyrics="Do not silently erase me"))

    def test_inline_limit_and_s3_long_song(self):
        with self.assertRaises(ValueError):
            parse_generate(request(duration_seconds=61))
        self.assertEqual(parse_generate(request(duration_seconds=240, output_mode="s3")).duration, 240)

    def test_outputs_seed_and_bpm_must_be_integer(self):
        for key, value in [("requested_outputs", 1.5), ("seed", 1.2), ("bpm", 120.5)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_generate(request(**{key: value}))

    def test_outputs_bounded(self):
        for value in [0, 3, "2", True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_generate(request(requested_outputs=value))

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            parse_generate(request(prompt="x" * 2401))

    def test_unknown_duration_spelling_rejected(self):
        with self.assertRaisesRegex(ValueError, "audio_duration"):
            validate_job({"input": request(audio_duration=15)})

    def test_arbitrary_path_model_and_url_rejected(self):
        for key in ["src_audio", "callback_url", "ace_model", "output_dir", "s3_endpoint"]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_job({"input": request(**{key: "/etc/passwd"})})

    def test_operation_contract(self):
        self.assertEqual(validate_job({"input": {"operation": "health"}})[0], "health")
        for job in [None, {}, {"input": []}, {"input": {"operation": "remix"}},
                    {"input": {"operation": "health", "prompt": "Bass"}}]:
            with self.subTest(job=job), self.assertRaises(ValueError):
                validate_job(job)

    def test_output_mode_is_validated(self):
        with self.assertRaises(ValueError):
            parse_generate(request(output_mode="local"))

    def test_private_voice_reference_is_accepted(self):
        token = "a" * 64
        parsed = parse_generate(request(
            reference_audio_url="https://zakul-ai.com/api/voice-reference/job-123",
            reference_audio_token=token,
        ))
        self.assertEqual(parsed.reference_audio_token, token)
        self.assertTrue(parsed.reference_audio_url.endswith("/job-123"))

    def test_reference_url_and_token_are_rejected_unless_both_are_safe(self):
        token = "a" * 64
        cases = [
            {"reference_audio_url": "https://zakul-ai.com/api/voice-reference/job"},
            {"reference_audio_token": token},
            {"reference_audio_url": "https://evil.example/api/voice-reference/job",
             "reference_audio_token": token},
            {"reference_audio_url": "https://zakul-ai.com/api/voice-reference/job?next=evil",
             "reference_audio_token": token},
            {"reference_audio_url": "https://zakul-ai.com/api/voice-reference/job",
             "reference_audio_token": "A" * 64},
        ]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                parse_generate(request(**fields))


if __name__ == "__main__":
    unittest.main()
