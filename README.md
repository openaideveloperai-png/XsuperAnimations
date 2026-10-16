# XsuperAnimations

**Make anime episodes from a prompt, a script, or a terrible doodle, with free, open AI models.**

You describe the story (or let an AI write it), XsuperAnimations turns it into a full episode script, draws every
shot with an anime model, keeps the characters on-model from the first shot to the last, gives them voices and
lip flaps, moves the camera, adds sakuga effects (speed lines, impact frames, shake, flashes), puts in AI
in-between animation for the key moments, and cuts it all into one video with sound and subtitles. Episodes can
be up to TV length (23 minutes).

```
idea ─► script (episode.yaml) ─► voices ─► keyframes ─► lip-flap & blink frames ─► AI motion clips
                                                     └─► camera moves + effects + colour grade ─► episode.mp4
```

> **About the look:** with the default art model (Animagine XL 4) and the `bright_fantasy` style, frames come
> out in the clean, modern TV-anime look: big shiny eyes, soft cel shading, bright colours. The
> `dark_action` style is grittier and more cinematic. Results depend on the model, the prompt and how much you
> curate (reroll bad shots). No tool turns a prompt into a hand-animated studio episode, but this one is
> built to fake it the way TV anime itself does.

---

## How it gets the "real anime" feel instead of "AI art"

Most TV anime isn't fully animated; it uses *limited animation*. XsuperAnimations does the same, and only uses AI
for the drawing:

| TV anime technique | What XsuperAnimations does |
|---|---|
| Model sheets keep characters on-model | Every character has fixed design tags, a fixed seed per shot, and optionally a **LoRA** trained on their model sheet |
| Held drawings with camera pans / zooms | Each shot is a high-res keyframe with smooth camera moves (pan, tilt, push-in, zoom punch, shake…) |
| 3-position mouth flaps, blinks | The mouth and eyes are repainted (open mouth / closed eyes) and swapped in time with the voice |
| Animating "on twos" | Mouth flaps, effects and AI motion clips change every 2 frames, which reads as hand-drawn rather than AI-smooth |
| Sakuga moments | `motion: ai` shots are animated with the Wan 2.2 video model, colour-matched to the keyframe |
| Compositing / satsuei | Per-style colour grade, bloom, vignette, grain, speed lines, impact frames, rain, letterbox |
| Shot grammar | Defaults follow anime editing: establishing shot → medium → close-ups → reactions |
| Prompting | `anime screencap, cel shading` style tags, with painting / 3D / realistic in the negative prompt |

---

## Quick start (free, no install): Google Colab

1. Open **[`colab/XsuperAnimations.ipynb`](colab/XsuperAnimations.ipynb)** in Google Colab
   (on GitHub, open the file, then use *Open in Colab*, or upload it at colab.research.google.com).
2. `Runtime → Change runtime type → T4 GPU`.
3. Run cells **1 → 4**. Cell 4 prints a public link to the studio app.

Projects are saved to your Google Drive. Rendering is cached, so if Colab disconnects you just run it again and it
carries on where it stopped. (Kaggle notebooks are a free alternative with more GPU hours per week.)

## Quick start: your own computer

You need Python 3.10+ and ffmpeg. For the AI art you need an NVIDIA GPU (8 GB+ VRAM; 12 GB+ for AI motion) running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI). Without a GPU you can still write episodes and preview them
as animatics.

```bash
git clone https://github.com/openaideveloperai-png/XsuperAnimations
cd XsuperAnimations
pip install -e ".[ui,voice]"          # voices also need espeak-ng installed

xsuper new my_show --example           # project with the demo episode "Ashfall"
xsuper render my_show                  # instant animatic preview (no GPU)
xsuper ui my_show                      # or use the web app
```

Put these model files in your ComfyUI folders (all free downloads; the Colab notebook shows exactly where they come from):

| File | Folder | Used for |
|---|---|---|
| `animagine-xl-4.0.safetensors` (or Illustrious / NoobAI) | `models/checkpoints` | all drawings |
| `controlnet-scribble-sdxl-1.0.safetensors` (xinsir) | `models/controlnet` | doodle → anime |
| `wan2.2_ti2v_5B_fp16.safetensors` | `models/diffusion_models` | AI motion shots |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders` | AI motion shots |
| `wan2.2_vae.safetensors` | `models/vae` | AI motion shots |
| `RealESRGAN_x4plus_anime_6B.pth` | `models/upscale_models` | sharpening AI motion |

Then `xsuper doctor my_show` checks everything, and `xsuper render my_show --mode final` makes the real thing.
If your files have other names, change them in `my_show/studio.yaml`.

---

## The workflow

1. **Story**: `xsuper write my_show "a girl who can rewind time 10 seconds joins a heist crew" --minutes 23`
   writes the whole episode. It uses any free LLM: [Ollama](https://ollama.com) on your PC (default), or a free
   API key from Groq / Google AI Studio / OpenRouter (set it in `studio.yaml → llm`). No LLM at all? Add
   `--prompt-only` and paste the prompt into any free chatbot.
2. **Script**: open `episode.yaml` and change anything you want (lines, shots, camera, effects). `xsuper check` validates it.
3. **Characters**: `xsuper sheet my_show ren` draws a model sheet. Delete off-model drawings, then (optionally)
   train a LoRA with the generated `train_lora.sh` to lock the design in.
4. **Animatic**: `xsuper render my_show` makes a stick-figure storyboard video in minutes, with no GPU needed.
   Use it to check pacing before spending GPU time.
5. **Final**: `xsuper render my_show --mode final`. Bad drawing? `--reroll shot_id` redraws only that shot.
   `--only id1,id2` renders a quick partial. `build/final/review.html` shows every keyframe side by side.

Modes: **animatic** (storyboard, no GPU) · **draft** (AI art, fewer steps, no AI motion; fast) · **final** (everything).

## episode.yaml cheat sheet

```yaml
title: Ashfall
style: bright_fantasy      # bright_fantasy dark_action bright_shonen soft_drama retro_90s noir_horror
language: en               # en or ja voices
seed: 4242                 # same seed = same drawings

characters:
  ren:
    name: Ren Kagari
    kind: boy              # boy girl man woman other
    tags: messy white hair, red eyes, small scar on left cheek      # design - never changes
    outfits:
      default: black gakuran, long red scarf
      battle: torn black gakuran, bandaged arms
    voice: am_michael      # optional (see Voices)
    lora: ren.safetensors  # optional, after training
    trigger: ren_xs

locations:
  rooftop: school rooftop, chain-link fence, city skyline in background

scenes:
  - id: rooftop
    location: rooftop
    time: sunset           # day morning sunset dusk night rain snow indoor indoor_night
    music: music/tense.mp3 # optional, loops and ducks under dialogue
    outfits: {ren: battle} # optional outfit change for the whole scene
    shots:
      - action: empty rooftop, wind           # establishing shot
        camera: pan_left
        transition: fade
      - say: ren                              # dialogue: close-up, voice, lip flaps, subtitle
        line: Go home, Mika.
        expression: sad
      - characters: [mika]                    # reaction shot while Ren talks off-screen
        say: ren
        line: It's not safe.
      - characters: [ren]
        action: jumping over the fence, midair, scarf flowing
        sketch: sketches/leap.png             # your doodle decides the pose
        sketch_strength: 0.6
      - characters: [ren]
        action: punching, red flames
        motion: ai                            # real AI in-between animation
        fx: [impact, speed_lines, shake]
      - card: "To be continued"
```

| Field | Values |
|---|---|
| `framing` | `extreme_closeup` `closeup` `medium` `cowboy` `full` `wide` `over_shoulder` `pov` |
| `angle` | `eye` `low` `high` `dutch` `side` `behind` `aerial` |
| `camera` | `auto` `hold` `drift` `pan_left` `pan_right` `tilt_up` `tilt_down` `push_in` `pull_out` `zoom_punch` `shake` |
| `fx` | `speed_lines` `streaks` `impact` `shake` `flash` `rain` `glow` `letterbox` |
| `expression` | `neutral` `serious` `happy` `laughing` `angry` `furious` `sad` `crying` `shocked` `scared` `smirk` `embarrassed` `determined` `tired` `crazy`, or any tags |
| `transition` | `cut` `fade` `flash` |
| other | `seconds`, `seed`, `outfit`, `sfx`, `sfx_volume`, `video` (insert your own clip, e.g. an OP), `card`, `face: [x, y, w, h]` |

## Keeping characters consistent

* **Design tags never change.** Only expression, outfit and pose vary per shot.
* **Seeds are fixed per shot**, so re-rendering gives the same drawing (until you `--reroll`).
* **Scene colour matching** pulls every shot in a scene toward one palette.
* **LoRA (best):** `xsuper sheet` → curate → `train_lora.sh` (free, ~1 hour on a Colab T4) → `lora:` + `trigger:`.
  This is how you get "same character, every episode".
* Two or more characters in one shot is the hardest case for any image model. Prefer close-ups and
  shot/reverse-shot for dialogue (anime does this anyway), and reroll group shots that mix designs.

## Doodle → anime

Draw anything, even stick figures, on paper (take a photo) or in any paint app. Use it with `sketch:` in a shot or in
the app's **Doodle → Anime** tab. `sketch_strength` 0.4-0.5 lets the AI fix your anatomy; 0.7-0.9 follows your lines
closely.

## Voices, music, OP/ED

Voices use [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (free, runs on CPU). English: `af_heart af_bella af_nicole
af_sarah af_sky af_nova bf_emma bf_isabella` (female) · `am_michael am_adam am_eric am_liam am_onyx am_puck bm_george
bm_lewis` (male). Japanese (`language: ja`, `pip install "misaki[ja]"`): `jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo`.
`voice_speed: 0.85` on a character makes them slower/heavier. Without Kokoro, dialogue is silent but subtitles still work.
You can also record your own voices and use them as `sfx`.

Music: `music:` per scene (any mp3/ogg/wav you're allowed to use). OP/ED: a shot with `video: op.mp4`.

## How long does it take?

These are rough estimates and vary a lot with GPU and settings.

* **Animatic** of a 23-min episode: about 30-60 minutes on a normal CPU.
* **Final keyframes:** about 1 minute each on a Colab T4 (much faster on an RTX 3060+). A 23-min episode has about 300-400 shots.
* **AI motion clips:** several minutes each on a T4. Use `motion: ai` for the 10-20 moments that matter, the way
  TV anime saves its sakuga for key scenes.
* Everything is cached. Render keyframes first (`--keyframes-only`), fix the bad ones with `--reroll`, then film.

## Honest limitations

* It needs a GPU for the real art: free Colab/Kaggle, or your own NVIDIA card.
* AI art still makes mistakes (hands, crowds, two characters mixing). The review sheet + `--reroll` is the fix, and
  a character LoRA makes a big difference.
* AI motion (Wan 2.2) is short (about 2 s per clip) and can drift. It is colour-matched and played on twos to stay on-style.
* Use the style presets to get *a look*. Don't use it to copy specific copyrighted characters, and only use
  music/voices you have the rights to.

## Project layout

```
my_show/
  episode.yaml      the whole episode (story, cast, shots)
  studio.yaml       models, ComfyUI address, LLM, encoder settings
  sketches/ music/ sfx/ characters/
  build/
    animatic/ draft/ final/      cached voices, keyframes, clips, shots, review.html
    <title>_ep01_final.mp4       the episode (+ .srt subtitles)
```

## Development

```bash
pip install -e ".[dev]" && pytest
```

The ComfyUI workflows use only built-in ComfyUI nodes and are checked against a real ComfyUI install.
Credits: [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Animagine XL](https://huggingface.co/cagliostrolab/animagine-xl-4.0),
[Wan 2.2](https://github.com/Wan-Video/Wan2.2), [xinsir ControlNet](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0),
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
[lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface), [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts).
Check each model's license before publishing what you make.
