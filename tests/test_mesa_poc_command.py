from __future__ import annotations

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rdc.build_mesa_poc_command import (  # noqa: E402
    CapsuleError,
    _bound_state_create_call,
    _current_constant_hex,
    _live_created_object_call,
    _pointer_array,
    _ret_text,
)


def parse_calls(body: str) -> list[ET.Element]:
    return ET.fromstring(f"<trace>{body}</trace>").findall("call")


class MesaPocCommandStateTests(unittest.TestCase):
    def test_pointer_array_preserves_null_slot_positions(self) -> None:
        argument = ET.fromstring(
            """
            <arg name="views"><array>
              <elem><null /></elem>
              <elem><ptr>view-b</ptr></elem>
            </array></arg>
            """
        )

        self.assertEqual(
            _pointer_array(argument, "sampler-view slots"), [None, "view-b"]
        )

    def test_object_lifecycle_rejects_destroyed_generation(self) -> None:
        calls = parse_calls(
            """
            <call method="create_sampler_view"><ret><ptr>view-a</ptr></ret></call>
            <call method="sampler_view_destroy">
              <arg name="view"><ptr>view-a</ptr></arg>
            </call>
            """
        )

        with self.assertRaises(CapsuleError):
            _live_created_object_call(
                calls,
                create_method="create_sampler_view",
                delete_method="sampler_view_destroy",
                delete_arg="view",
                object_id="view-a",
                label="sampler-view",
            )

    def test_bound_state_ignores_later_unbound_creation(self) -> None:
        calls = parse_calls(
            """
            <call method="create_blend_state"><ret><ptr>state-a</ptr></ret></call>
            <call method="bind_blend_state">
              <arg name="state"><ptr>state-a</ptr></arg>
            </call>
            <call method="create_blend_state"><ret><ptr>state-b</ptr></ret></call>
            """
        )

        resolved = _bound_state_create_call(
            calls,
            create_method="create_blend_state",
            bind_method="bind_blend_state",
            label="blend state",
        )

        self.assertEqual(_ret_text(resolved, "ptr"), "state-a")

    def test_bound_state_rejects_deleted_binding(self) -> None:
        calls = parse_calls(
            """
            <call method="create_blend_state"><ret><ptr>state-a</ptr></ret></call>
            <call method="bind_blend_state">
              <arg name="state"><ptr>state-a</ptr></arg>
            </call>
            <call method="delete_blend_state">
              <arg name="state"><ptr>state-a</ptr></arg>
            </call>
            """
        )

        with self.assertRaises(CapsuleError):
            _bound_state_create_call(
                calls,
                create_method="create_blend_state",
                bind_method="bind_blend_state",
                label="blend state",
            )

    def test_constant_uses_current_binding_not_history(self) -> None:
        calls = parse_calls(
            """
            <call method="set_constant_buffer">
              <arg name="shader"><enum>MESA_SHADER_VERTEX</enum></arg>
              <arg name="index"><uint>0</uint></arg>
              <arg name="constant_buffer"><struct>
                <member name="buffer"><ptr>buffer-a</ptr></member>
                <member name="buffer_size"><uint>4</uint></member>
              </struct></arg>
              <arg name="data"><bytes>AAAAAAAA</bytes></arg>
            </call>
            <call method="set_constant_buffer">
              <arg name="shader"><enum>MESA_SHADER_VERTEX</enum></arg>
              <arg name="index"><uint>0</uint></arg>
              <arg name="constant_buffer"><struct>
                <member name="buffer"><ptr>buffer-b</ptr></member>
                <member name="buffer_size"><uint>4</uint></member>
              </struct></arg>
              <arg name="data"><bytes>bbbbbbbb</bytes></arg>
            </call>
            """
        )

        self.assertEqual(
            _current_constant_hex(calls, "MESA_SHADER_VERTEX"), "BBBBBBBB"
        )

    def test_constant_null_binding_clears_previous_bytes(self) -> None:
        calls = parse_calls(
            """
            <call method="set_constant_buffer">
              <arg name="shader"><enum>MESA_SHADER_FRAGMENT</enum></arg>
              <arg name="index"><uint>0</uint></arg>
              <arg name="constant_buffer"><struct>
                <member name="buffer"><ptr>buffer-a</ptr></member>
                <member name="buffer_size"><uint>4</uint></member>
              </struct></arg>
              <arg name="data"><bytes>AAAAAAAA</bytes></arg>
            </call>
            <call method="set_constant_buffer">
              <arg name="shader"><enum>MESA_SHADER_FRAGMENT</enum></arg>
              <arg name="index"><uint>0</uint></arg>
              <arg name="constant_buffer"><null /></arg>
            </call>
            """
        )

        self.assertEqual(
            _current_constant_hex(calls, "MESA_SHADER_FRAGMENT"), "NONE"
        )


if __name__ == "__main__":
    unittest.main()
