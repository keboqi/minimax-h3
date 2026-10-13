"""Protocol failure cases for execution on the processing host/CI only."""
import io
import struct
import unittest

from h3_app import dlss_worker as worker


class DlssTransportTests(unittest.TestCase):
    def test_index_and_payload_length_are_checked_before_reading_pixels(self):
        for header in ((b"logs", 0, 1, 3), (b"OUT1", 1, 1, 3), (b"OUT1", 0, 1, 99)):
            with self.subTest(header=header), self.assertRaises(RuntimeError):
                worker.decode_reply(io.BytesIO(worker.REPLY.pack(*header)), 0, 3)

    def test_actual_engine_error_is_retained(self):
        message = b"NGX core initialization failed: 0xBAD00001"
        reply = worker.REPLY.pack(b"OUT1", 0, 0, 0) + struct.pack("<I", len(message)) + message
        with self.assertRaisesRegex(RuntimeError, "NGX core initialization failed"):
            worker.decode_reply(io.BytesIO(reply), 0, 3)

    def test_complete_and_truncated_float_replies(self):
        payload = struct.pack("<3f", 0.1, 0.2, 0.3)
        header = worker.REPLY.pack(b"OUT1", 7, 1, 3)
        self.assertEqual(worker.decode_reply(io.BytesIO(header + payload), 7, 3), payload)
        with self.assertRaisesRegex(RuntimeError, "expected bytes"):
            worker.decode_reply(io.BytesIO(header + payload[:-1]), 7, 3)

    def test_dnr2_layout_matches_the_native_host(self):
        self.assertEqual(worker.HEADER.size, 72)
        self.assertEqual(worker.FRAME.size, 12)
        self.assertEqual(worker.REPLY.size, 16)


if __name__ == "__main__":
    unittest.main()
