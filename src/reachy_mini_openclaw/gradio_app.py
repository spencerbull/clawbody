"""Gradio web UI for Reachy Mini OpenClaw.

This module provides a web interface for:
- Viewing conversation transcripts
- Configuring the assistant personality
- Monitoring robot status
- Manual control options
- Browser audio routing (mic / speaker via WebRTC instead of robot hardware)
"""

import asyncio
import logging
import threading
from typing import Optional

import gradio as gr
from fastrtc import WebRTC

from reachy_mini_openclaw.browser_audio import BrowserAudioBridge

logger = logging.getLogger(__name__)


def launch_gradio(
    gateway_url: str = "ws://localhost:18789",
    robot_name: Optional[str] = None,
    enable_camera: bool = True,
    enable_openclaw: bool = True,
    enable_face_tracking: bool = True,
    head_tracker_type: Optional[str] = None,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI.

    Args:
        gateway_url: OpenClaw gateway URL
        robot_name: Robot name for connection
        enable_camera: Whether to enable camera
        enable_openclaw: Whether to enable OpenClaw
        enable_face_tracking: Whether to enable face tracking
        head_tracker_type: Head tracker type ('yolo', 'mediapipe', or None)
        share: Whether to create a public URL
    """
    from reachy_mini_openclaw.prompts import get_available_profiles, save_custom_profile
    from reachy_mini_openclaw.config import set_custom_profile, config

    # -----------------------------------------------------------------------
    # Browser audio bridge — created once here, before any "Start" click.
    # ClawBodyCore receives it when the user starts a session and calls
    # bridge.attach_handler(core.handler) to wire it to the live pipeline.
    # -----------------------------------------------------------------------
    bridge = BrowserAudioBridge()

    # Session state
    app_instance = None

    # -----------------------------------------------------------------------
    # Session control callbacks
    # -----------------------------------------------------------------------

    def start_conversation():
        """Start the conversation."""
        nonlocal app_instance

        from reachy_mini_openclaw.main import ClawBodyCore

        if app_instance is not None:
            return "Already running"

        try:
            from reachy_mini_openclaw.config import set_face_tracking_enabled, set_head_tracker_type

            if not enable_face_tracking:
                set_face_tracking_enabled(False)
            if head_tracker_type is not None:
                set_head_tracker_type(head_tracker_type)

            app_instance = ClawBodyCore(
                gateway_url=gateway_url,
                robot_name=robot_name,
                enable_camera=enable_camera,
                enable_openclaw=enable_openclaw,
                browser_bridge=bridge,
            )

            # Run in background thread
            def run_app():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(app_instance.run())
                except Exception as e:
                    logger.error("App error: %s", e)
                finally:
                    loop.close()

            thread = threading.Thread(target=run_app, daemon=True)
            thread.start()

            return "Started successfully"
        except Exception as e:
            return f"Error: {e}"

    def stop_conversation():
        """Stop the conversation."""
        nonlocal app_instance

        if app_instance is None:
            return "Not running"

        try:
            app_instance.stop()
            app_instance = None
            return "Stopped"
        except Exception as e:
            return f"Error: {e}"

    # -----------------------------------------------------------------------
    # Browser audio routing callbacks (live toggle — no restart needed)
    # -----------------------------------------------------------------------

    def set_browser_mic(enabled: bool):
        """Toggle browser mic routing on/off."""
        bridge.routing.use_browser_mic = enabled
        state = "ON — browser mic active, robot mic paused" if enabled else "OFF — robot mic active"
        logger.info("Browser mic: %s", state)

    def set_browser_speaker(enabled: bool):
        """Toggle browser speaker routing on/off."""
        bridge.routing.use_browser_speaker = enabled
        state = "ON — browser speaker active, robot speaker paused" if enabled else "OFF — robot speaker active"
        logger.info("Browser speaker: %s", state)

    # -----------------------------------------------------------------------
    # Personality / profile callbacks
    # -----------------------------------------------------------------------

    def apply_profile(profile_name):
        """Apply a personality profile."""
        set_custom_profile(profile_name if profile_name else None)
        return f"Applied profile: {profile_name or 'default'}"

    def save_profile(name, instructions):
        """Save a new profile."""
        if save_custom_profile(name, instructions):
            return f"Saved profile: {name}"
        return "Error saving profile"

    # -----------------------------------------------------------------------
    # Build UI
    # -----------------------------------------------------------------------
    with gr.Blocks(title="Reachy Mini OpenClaw") as demo:
        gr.Markdown("# Reachy Mini OpenClaw")

        with gr.Tab("Conversation"):
            # Session control
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop", variant="secondary")
            status_text = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")

            # Browser audio section
            gr.Markdown("### Browser Audio")
            gr.Markdown(
                "Use your browser's microphone and/or speaker instead of the robot's "
                "hardware. Switches are live — you can toggle them mid-conversation without "
                "restarting. Connect the WebRTC session below before enabling the toggles."
            )

            with gr.Row():
                browser_mic_toggle = gr.Checkbox(
                    label="Browser Mic",
                    value=False,
                    info="ON: browser mic → speaches  |  OFF: robot mic → speaches",
                )
                browser_speaker_toggle = gr.Checkbox(
                    label="Browser Speaker",
                    value=False,
                    info="ON: speaches audio → browser  |  OFF: speaches audio → robot speaker",
                )

            # WebRTC component — always rendered so the browser can establish
            # an ICE connection before the session starts.  Audio only flows
            # when the toggles above are ON.
            webrtc = WebRTC(
                label="Browser Audio (WebRTC)",
                mode="send-receive",
                modality="audio",
                # full_screen=False keeps the component inline.
                # The default (True) renders a full-screen overlay that blocks all other UI.
                full_screen=False,
                # No STUN/TURN configured — works for local LAN access.
                # For remote/internet access add:
                #   rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
                rtc_configuration=None,
            )

            # Wire the WebRTC component to the bridge handler
            webrtc.stream(
                fn=bridge,
                inputs=[webrtc],
                outputs=[webrtc],
            )

            gr.Markdown("---")

            # Transcript
            transcript = gr.Chatbot(label="Conversation", height=400)

            # Wire session control buttons
            start_btn.click(start_conversation, outputs=[status_text])
            stop_btn.click(stop_conversation, outputs=[status_text])

            # Wire browser audio toggles (live, no session restart)
            browser_mic_toggle.change(set_browser_mic, inputs=[browser_mic_toggle])
            browser_speaker_toggle.change(set_browser_speaker, inputs=[browser_speaker_toggle])

        with gr.Tab("Personality"):
            profiles = get_available_profiles()
            profile_dropdown = gr.Dropdown(choices=[""] + profiles, label="Select Profile", value="")
            apply_btn = gr.Button("Apply Profile")
            profile_status = gr.Textbox(label="Status", interactive=False)

            apply_btn.click(apply_profile, inputs=[profile_dropdown], outputs=[profile_status])

            gr.Markdown("### Create New Profile")
            new_name = gr.Textbox(label="Profile Name")
            new_instructions = gr.Textbox(
                label="Instructions", lines=10, placeholder="Enter the system prompt for this personality..."
            )
            save_btn = gr.Button("Save Profile")
            save_status = gr.Textbox(label="Save Status", interactive=False)

            save_btn.click(save_profile, inputs=[new_name, new_instructions], outputs=[save_status])

        with gr.Tab("Settings"):
            gr.Markdown(f"""
            ### Current Configuration

            - **OpenClaw Gateway**: {gateway_url}
            - **Speaches Model**: {config.SPEACHES_REALTIME_MODEL}
            - **Voice**: {config.SPEACHES_VOICE}
            - **STT Model**: {config.SPEACHES_STT_MODEL}
            - **TTS Model**: {config.SPEACHES_TTS_MODEL}
            - **Camera Enabled**: {enable_camera}
            - **OpenClaw Enabled**: {enable_openclaw}
            - **Face Tracking**: {enable_face_tracking}
            - **Head Tracker**: {head_tracker_type or "auto-detect"}

            Edit `.envrc` to change these settings.
            """)

        with gr.Tab("About"):
            gr.Markdown("""
            ## About Reachy Mini OpenClaw

            This application combines:

            - **OpenAI Realtime API** for ultra-low-latency voice conversation
            - **OpenClaw Gateway** for extended AI capabilities (web, calendar, smart home, etc.)
            - **Reachy Mini Robot** for physical embodiment with expressive movements

            ### Browser Audio

            The **Browser Audio** section in the Conversation tab lets you route
            microphone input and speaker output through your browser instead of
            (or switching back to) the robot's hardware:

            - **Browser Mic ON**: your laptop/desktop mic speaks to the AI instead of
              the robot's mic. The browser handles echo cancellation natively.
            - **Browser Speaker ON**: the AI's voice plays through your browser speaker
              instead of the robot's speaker.

            Both toggles are independent and live — flip them mid-conversation.

            ### Features

            - Real-time voice conversation
            - Camera-based vision
            - Expressive robot movements
            - Tool integration via OpenClaw
            - Customizable personalities
            """)

    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)
