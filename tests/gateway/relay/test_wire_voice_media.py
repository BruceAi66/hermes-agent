"""Unit tests for relay voice-note + per-attachment media-type mapping.

The relay contract gained ``message_type: "voice"`` (connector PR: voice
notes on Discord/Telegram/WhatsApp classify distinctly from audio-file
uploads). The gateway side has two obligations:

1. ``MessageType("voice")`` must parse — it already does, since the enum
   predates the wire value — and
2. the connector's rich ``media[]`` array (each entry carries the
   per-attachment ``mime``) must land on ``event.media_types`` so run.py's
   per-attachment classifiers (``_event_media_is_stt_input``, image vs
   document routing) work for relayed events, not just native adapters.

Live-verified failure (staging 2026-08-26): a voice note arrived as
``MessageType.AUDIO`` with ``media_types == []`` — STT never fired and the
agent got a "the user sent an audio file attachment" note instead.

Pure unit tests: no socket, no websockets dependency.
"""

from __future__ import annotations

from gateway.platforms.base import MessageType
from gateway.relay.ws_transport import _event_from_wire


def _wire_event(message_type: str, **extra):
    src = {
        "platform": "discord",
        "chat_id": "chan-1",
        "chat_type": "dm",
        "user_id": "u-1",
        "user_name": "ben",
    }
    return {
        "text": "",
        "message_type": message_type,
        "source": src,
        **extra,
    }


class TestVoiceMessageType:

    def test_wire_voice_parses_to_message_type_voice(self):
        ev = _event_from_wire(_wire_event("voice"))
        assert ev.message_type == MessageType.VOICE

    def test_wire_audio_still_parses_to_message_type_audio(self):
        """Music files keep the non-STT type — the connector must not have
        collapsed them (it doesn't), and the gateway must not either."""
        ev = _event_from_wire(_wire_event("audio"))
        assert ev.message_type == MessageType.AUDIO


class TestMediaTypesMapping:

    def test_media_mimes_land_on_event_media_types(self):
        media = [
            {
                "url": "https://cdn.discordapp.com/attachments/1/2/voice-message.ogg",
                "kind": "audio",
                "mime": "audio/ogg",
                "size": 19197,
                "filename": "voice-message.ogg",
            },
        ]
        ev = _event_from_wire(
            _wire_event("voice", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["audio/ogg"]
        # media_urls must remain the parallel legacy field (unchanged).
        assert ev.media_urls == [media[0]["url"]]

    def test_media_types_parallel_to_media_urls_for_multiple_attachments(self):
        media = [
            {"url": "https://x/photo.png", "kind": "image", "mime": "image/png"},
            {"url": "https://x/doc.pdf", "kind": "document", "mime": "application/pdf"},
        ]
        ev = _event_from_wire(
            _wire_event("image", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["image/png", "application/pdf"]
        assert len(ev.media_types) == len(ev.media_urls)

    def test_missing_mime_gives_empty_string_at_that_index(self):
        """A media entry without a mime must not shift the alignment between
        media_urls[i] and media_types[i] — run.py indexes both by position."""
        media = [
            {"url": "https://x/photo.png", "kind": "image", "mime": "image/png"},
            {"url": "https://x/blob", "kind": "document"},  # no mime
        ]
        ev = _event_from_wire(
            _wire_event("image", media=media, media_urls=[m["url"] for m in media])
        )
        assert ev.media_types == ["image/png", ""]

    def test_no_media_field_means_empty_media_types(self):
        """Older connectors (no media[]) — byte-identical to pre-fix."""
        ev = _event_from_wire(_wire_event("text"))
        assert ev.media_types == []
