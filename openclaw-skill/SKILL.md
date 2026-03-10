---
name: clawbody
description: Give your OpenClaw AI agent a physical robot body with Reachy Mini. Works with physical robot OR simulator! Voice conversation powered by speaches (local STT + TTS) with OpenClaw as the primary intelligence.
---

# ClawBody - Robot Body for OpenClaw

Give your OpenClaw agent (Clawson) a physical robot body with Reachy Mini.

## Overview

ClawBody embodies your OpenClaw AI assistant in a Reachy Mini robot, enabling it to:

- **Hear**: Listen to voice commands via the robot's microphone
- **See**: View the world through the robot's camera
- **Speak**: Respond with natural voice through the robot's speaker
- **Move**: Express emotions through expressive head movements and dances

OpenClaw is the **primary intelligence** — all conversation, memory, and tool use flows through the OpenClaw gateway. speaches handles local STT (Whisper) and TTS (Kokoro) only.

## Architecture

```
You speak → Reachy Mini mic
                ↓
        speaches (local)
      VAD + STT (Whisper)
                ↓
        OpenClaw Gateway
      (Clawson's brain — primary AI)
                ↓ (text response)
        speaches HTTP TTS
           (Kokoro)
                ↓
   Robot speaks & moves
```

## Robot Tool Webhooks

OpenClaw controls robot movements by calling webhook endpoints on the RobotToolServer (port 8234 by default). Configure the following tools in your OpenClaw agent to point at `http://<robot-ip>:8234/tools/<name>`:

| Tool | Endpoint | Description |
|------|----------|-------------|
| `look` | `POST http://localhost:8234/tools/look` | Move robot head to look at a direction |
| `emotion` | `POST http://localhost:8234/tools/emotion` | Express an emotion (happy, curious, thinking, excited) |
| `dance` | `POST http://localhost:8234/tools/dance` | Perform a dance animation |
| `camera` | `POST http://localhost:8234/tools/camera` | Capture a photo from the robot camera |
| `face_tracking` | `POST http://localhost:8234/tools/face_tracking` | Enable/disable face tracking |
| `stop_moves` | `POST http://localhost:8234/tools/stop_moves` | Stop all current movements |
| `idle` | `POST http://localhost:8234/tools/idle` | Enter idle/resting state |

Health check: `GET http://localhost:8234/health`

## Requirements

### Option A: Physical Robot
- [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot (Wireless or Lite)

### Option B: Simulator (No Hardware Required!)
- Any computer with Python 3.11+
- Install: `pip install "reachy-mini[mujoco]"`
- [Simulator Setup Guide](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)

### Software (Both Options)
- Python 3.11+
- OpenAI API key with Realtime API access
- OpenClaw gateway running on your network

## Installation

```bash
# Clone from GitHub
git clone https://github.com/tomrikert/clawbody
cd clawbody
pip install -e .
```

Or from HuggingFace:
```bash
git clone https://huggingface.co/spaces/tomrikert/clawbody
```

## Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENCLAW_GATEWAY_URL=http://your-host-ip:18789
OPENCLAW_TOKEN=your-gateway-token
```

## Usage

### With Simulator (No Robot Required)

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Run ClawBody
clawbody --gradio
```

### With Physical Robot

```bash
# Run ClawBody
clawbody

# With debug logging
clawbody --debug

# With Gradio web UI
clawbody --gradio
```

## Features

### Real-time Voice Conversation
Ultra-low latency voice interaction using OpenAI's Realtime API for speech recognition and text-to-speech.

### OpenClaw Intelligence
Full Clawson capabilities - tools, memory, personality - through the OpenClaw gateway HTTP API.

### Expressive Movements
- Audio-driven head wobble while speaking
- Emotion expressions (happy, curious, thinking, excited)
- Dance animations
- Natural head tracking

### Vision
Ask Clawson to describe what it sees through the robot's camera.

## Links

- [GitHub Repository](https://github.com/tomrikert/clawbody)
- [HuggingFace Space](https://huggingface.co/spaces/tomrikert/clawbody)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Documentation](https://docs.openclaw.ai)

## Author

Tom (tomrikert)

## License

Apache 2.0
