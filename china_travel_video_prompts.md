# AI Video Prompting for China Travel Content

**Focus:** Prompt engineering for AI video generation tools (Kling, Wan, Sora, Runway, Veo, Seedance) applied to China travel and cultural content.

---

## 1. Universal Prompt Structure

### The 7-Layer Framework
```
[Shot type] + [Subject/detail] + [Action/motion] +
[Setting with 3–5 sensory details] + [Camera movement] +
[Lighting/mood] + [Style/aesthetic] + [Negative prompts]
```

### Key Rules
- One camera movement per clip — combining complex moves warps geometry
- Layer **foreground / midground / background** explicitly
- Use exact motion verbs: *glides, drifts, ascends, arcs* — not "moves"
- Keep element count to 3–7 items depending on model
- Always specify an **endpoint** for motion to prevent generation hangs
- Use temporal markers for progression: "initially... then... finally"

---

## 2. Location-Specific Prompts

### The Great Wall

**Cinematic drone reveal:**
```
Aerial drone shot ascending over the Great Wall of China at dawn, revealing
watchtowers stretching across mountain ridges into morning mist. Camera rises
vertically then arcs forward in a slow crane movement. Soft golden backlight
silhouettes the crenellations. Volumetric fog fills the valleys below.
24mm anamorphic lens with lens flare. Cinematic color grade, teal shadows,
warm highlights.
No modern structures, no crowds, no text.
```

**FPV approach (Kling-optimized):**
```
The camera zooms into a beacon tower on the Great Wall, first-person
perspective, high-speed flight, symmetrical composition, motion blur,
atmospheric lighting.
```

**Autumn wide shot:**
```
Wide establishing shot of the Mutianyu section of the Great Wall in autumn,
orange and gold forest canopy covering hillsides, lone figure in traditional
clothing walking the ramparts. Camera slowly dollies forward from low angle.
Late afternoon warm sidelight. Anamorphic, film grain, National Geographic
documentary style.
No tourists, no signs.
```

> **Tip:** Specify section name ("Jiankou," "Mutianyu," "Simatai") — without it, models default to the Badaling tourist context.

**Negative prompts:** `modern buildings, crowds, tourists, road signs, power lines, Western architecture`

---

### Zhangjiajie (Avatar Mountains)

**Aerial mist reveal:**
```
Aerial drone shot slowly ascending through morning mist in Zhangjiajie
National Forest Park, revealing towering sandstone pillar formations emerging
above the cloud layer. Slow forward dolly through the mist, pillars appearing
and disappearing. Ancient hanging bridge in midground. Soft diffused morning
light. Lush green subtropical forest covering column bases. Cinematic
wide-angle, deep depth of field, National Geographic style.
No modern structures, no cable cars.
```

**Golden hour tracking:**
```
Tracking shot along a narrow mountain path winding between the Avatar
Hallelujah Mountains at golden hour, vertical pillars glowing amber,
wisps of cloud drifting at mid-height. Camera tracks forward at eye level
then cranes upward to reveal full scale. Atmospheric haze, volumetric light
rays. Epic fantasy cinematography. Anamorphic widescreen.
```

> **Tip:** Use "Zhangjiajie sandstone pillars" — generic "floating mountains" risks fantasy/sci-fi output.

---

### Li River / Guilin

**Classic river sunrise:**
```
Wide cinematic shot of a traditional bamboo raft drifting on the Li River
at sunrise, limestone karst peaks reflected in still water, morning mist
clinging to the mountain bases. Camera slowly trucks left to right, keeping
the raft centered. Soft pink and gold sunrise palette. Shallow depth of field
with peaks soft in distance. Documentary nature cinematography, 35mm film look.
No motor boats, no modern structures.
```

**Aerial karst reveal:**
```
Bird's-eye drone shot descending over the Li River valley near Yangshuo,
rows of jagged limestone peaks in receding layers, green rice fields and
winding river below. Slow orbital arc revealing scale. Overcast soft light,
rich greens, misty atmosphere. Cinematic 4K.
```

---

### Shanghai / The Bund

**Blue hour Bund sweep:**
```
Tracking shot moving slowly along the Bund waterfront at blue hour, historic
Art Deco colonial buildings on the left, Pudong skyline with Oriental Pearl
Tower and Shanghai Tower across the Huangpu River on the right. Light
reflections shimmer in water. Camera dollies forward at water level,
slight low angle. Moody blue-gold color palette, cinematic grade.
No crowds visible.
```

**City-to-night time-lapse:**
```
Time-lapse aerial shot of Shanghai Pudong at dusk transitioning to night,
skyline lighting up tower by tower, Huangpu River alive with vessel light
trails, clouds moving rapidly. Camera rises vertically during transition.
Cinematic wide-angle, deep depth of field, vivid neon and deep blue tones.
```

---

### Beijing — Forbidden City and Hutongs

**Forbidden City aerial:**
```
Drone shot ascending vertically above the Meridian Gate, revealing the vast
red-walled imperial compound receding in perfect symmetry toward Tiananmen
Square. Late autumn afternoon golden hour, yellow-glazed roof tiles glowing.
Slow vertical crane transitioning to a forward dolly over rooftops.
Deep cinematic color grade, warm ochre and crimson palette.
No crowds, no modern structures outside the walls.
```

> **Tip:** Explicitly state "yellow glazed ceramic tiles" and "red lacquered columns" — models default to generic Asian temple aesthetics with wrong colors.

**Hutong street life at dusk:**
```
Handheld tracking shot following a bicycle through a Beijing hutong at dusk,
grey brick courtyard walls on both sides, paper lanterns beginning to glow,
vendor selling jianbing at a corner stall. Camera follows slightly behind and
to the side at low angle. Warm tungsten practical lights, soft street ambient.
35mm documentary style, slight grain, shallow depth of field.
No modern signage, no foreign branding.
```

---

### Yunnan — Rice Terraces and Lijiang

**Hani rice terraces (Yuanyang) at dawn:**
```
Aerial crane shot slowly ascending over the Hani rice terraces of Yuanyang
at dawn, flooded terrace paddies reflecting the pink sunrise sky, terraces
cascading down steep hillsides in concentric curves. Camera rises from valley
floor, revealing more terraces ascending to the ridge. Fog fills lower
valleys. Warm pink-gold sunrise light. National Geographic style.
No buildings visible.
```

**Lijiang Old Town:**
```
Tracking shot through ancient streets of Lijiang Old Town at golden hour,
traditional Naxi timber-frame buildings with curved eaves lining narrow
cobblestone lanes, clear mountain stream running alongside, potted flowers
on doorsteps. Camera moves forward at walking pace. Warm afternoon sidelight.
Documentary travel style, 35mm, slight grain.
```

> **Tip:** Specify "Naxi architecture" not "traditional Chinese" — the timber-frame Lijiang style is ethnically distinct from Han architecture.

---

### Chengdu — Pandas and Teahouses

**Giant panda close-up:**
```
Medium shot of a giant panda sitting in a eucalyptus tree at Chengdu Research
Base, lazily chewing bamboo, occasional glance at camera. Camera slowly
pushes in from wide to medium close-up over 10 seconds. Soft dappled forest
light, green canopy background. Wildlife documentary, 85mm telephoto
simulation, shallow depth of field.
No fences, no signage, no people.
```

**Sichuan teahouse:**
```
Wide shot of a traditional Sichuan teahouse in People's Park, elderly patrons
playing mahjong and sipping tea from gaiwan cups beneath bamboo umbrellas.
Ear-cleaner moves between tables. Camera slowly pans left to right. Warm
afternoon diffused light filtering through bamboo. Documentary cinéma vérité
style, slight handheld movement, 35mm grain.
```

---

### Tibet / Potala Palace

**Palace aerial:**
```
Aerial view of the Potala Palace in Lhasa, its white and crimson fortress
rising from Marpo Ri hill against snow-capped Himalayan peaks. Early morning
light, gentle golden glow, soft mist below. Camera performs slow orbital arc
east to west. Wide-angle cinematic framing, teal-and-orange grade.
```

> **Tip:** Specify "Tibetan rammed-earth and stone fortress architecture, white lower Potrang and red upper Potrang" — prevents generic castle generation.

**Prayer flags:**
```
Close-up tracking shot following Tibetan prayer flags fluttering in wind on
a mountain pass, five-color flags filling the frame, blurred Himalayan peaks
behind. Camera slowly pulls back to reveal landscape below. High-altitude
harsh directional sunlight. Documentary style, natural color grade.
```

---

### Xi'an — Terracotta Army

**Underground pit reveal:**
```
Slow dolly forward into Pit 1 of the Terracotta Army, thousands of life-size
clay warriors in formation stretching into the distance under the protective
hangar. Camera moves at low angle between the rows. Dramatic overhead
industrial lighting casting deep shadows. Cinematic documentary, shallow
depth of field, warm terracotta palette.
No modern tourists, no glass barriers visible.
```

**Warrior close-up:**
```
Slow push-in on a Terracotta Army warrior face, revealing intricate
individual facial details — unique expression, painted traces on clay surface.
Camera begins wide at chest level, craning upward and closing to extreme
close-up. Museum side-lighting, dramatic single-source illumination.
Archaeological documentary aesthetic.
```

---

### Hong Kong

**Victoria Peak aerial:**
```
Aerial drone shot from Victoria Peak descending toward Hong Kong skyline at
twilight, skyscrapers glittering below, Victoria Harbour busy with vessels,
Kowloon on the far shore. Camera descends and arcs forward. Vivid city lights
against deep blue sky. Cyan and amber palette. Wide-angle, crisp depth.
No fog.
```

**Mong Kok night:**
```
Handheld tracking shot through Mong Kok at night, neon signs in Chinese
characters reflecting in wet pavement, dense pedestrian crowds, street food
stalls steaming, double-decker trams passing. Camera weaves through crowd at
street level. Warm neon reds and yellows. Documentary urban cinematography,
slight handheld shake, 35mm grain.
```

---

## 3. Festival and Cultural Prompts

**Spring Festival (Chinese New Year) street:**
```
Wide tracking shot through a village street decorated for Chinese New Year,
red lanterns strung between traditional buildings, children in new clothes,
firecrackers remnants on ground, gold and red banners. Camera moves forward
at eye level. Warm tungsten and lantern light. Documentary style.
```

**Lantern Festival river release:**
```
Low-angle wide shot of hundreds of sky lanterns being released above a river
at night, glowing orange and gold ascending into dark sky, reflections
rippling in water. Camera slowly tilts upward from river surface to follow
rising lanterns. Long exposure feel, warm glow, deep black sky.
```

**Dragon Boat Festival:**
```
Tracking shot following a dragon boat from a low lateral angle, paddlers in
unison, drummer pounding at the prow, water spraying, crowd cheering on the
bank. Fast-paced camera tracking at water level, slight handheld. Golden
afternoon light. Energetic, kinetic mood.
```

---

## 4. Style and Mood Vocabulary

| Mood | Prompt Language |
|---|---|
| Misty mountains | `morning mist filling valleys, volumetric fog, atmospheric haze, clouds at mid-height` |
| Golden hour | `golden hour sidelight, warm amber backlight, long shadows, anamorphic lens flares` |
| Dramatic grandeur | `epic wide-angle reveal, sweeping crane shot, IMAX-scale framing` |
| Documentary realism | `35mm grain, handheld sway, natural light, cinéma vérité, shallow depth of field` |
| Ink painting | `ink wash aesthetic, monochrome haze, minimalist composition, mountains fading to grey` |
| Festive | `warm tungsten practical light, lantern glow, bokeh from string lights, crowd movement` |

---

## 5. Camera Motion Reference

| Move | Prompt Language | Best Use |
|---|---|---|
| Crane | `camera crane upward while panning to reveal landscape` | Landscape reveals, temples |
| Dolly forward | `slow dolly in`, `camera pushes forward` | Intimacy, approaching subjects |
| Aerial drone | `aerial drone shot ascending`, `drone flyover` | Establishing shots, scale |
| Orbital arc | `smooth orbital arc around the subject` | Monuments, architecture |
| Tracking | `camera tracks alongside`, `lateral tracking shot` | Street scenes, moving subjects |
| Tilt reveal | `camera tilts upward revealing the full height of` | Tall buildings, mountain peaks |
| Pedestal rise | `camera rises vertically while maintaining framing` | City skylines, terraces |
| FPV dive | `FPV drone shot, high-speed dive and pull-up over` | Dynamic landscape passes |
| Handheld | `slight handheld sway`, `naturalistic handheld movement` | Markets, street life, vérité |

---

## 6. Tool-Specific Tips

### Kling (Kuaishou) — Best for Chinese Content
- **Why:** Trained on Kuaishou's massive Chinese short-video platform — more Chinese architecture, landscape, dance, and street-life content than any Western model
- **Best for:** Hutongs, traditional courtyard houses, classical dance, festivals, tea ceremonies, Great Wall drone work
- **Negative prompts:** Use the separate negative field; write just "X" (not "no X")
- **Element limits:** Kling 1.6: 3–4 elements; Kling 2.6+: 5–7 elements; Kling 3.0: full complexity
- **Universal Chinese negatives:** `western architecture, european facades, mixed asian styles`

### Wan 2.2 (Alibaba) — Best Cultural Nuance, Open Source
- **Why:** Accepts Chinese-language prompts natively — bypasses the translation layer that causes Western models to drift
- **Best for:** Specific regional architecture (Naxi in Lijiang, Tibetan in Lhasa, Dong minority villages)
- **Key technique:** Write location in Chinese (Simplified), add English cinematography terms:
```
张家界国家森林公园，清晨薄雾中的石英砂岩柱峰群，aerial drone shot
ascending through morning mist, cinematic wide-angle, golden hour lighting.
```

### Sora 2 (OpenAI) — Best Narrative Coherence, Weakest China Specificity
- **Best for:** Contemporary urban China (Shanghai Bund, Hong Kong neon, Chengdu modern), atmospheric landscape
- **Weakness:** Western-biased training; traditional Chinese architecture requires heavy explicit prompting to prevent European-style drift
- **Technique:** Shot Stack structure — break scene into temporal beats: OPENING → BEAT 1 → BEAT 2 → CLOSING

### Runway Gen-4 — Best Camera Control + Image-to-Video
- **Key technique:** Use image-to-video with reference photos of the actual location to lock in architectural accuracy
- **Camera vocabulary:** Use Runway's exact terms: `Dolly`, `Truck`, `Pedestal`, `Orbit`, `Crane shot`, `Arc shot`, `Whip pan`
- **Style descriptors:** `Cinematic`, `Smooth Animation`, `Vintage`, `Live-action`

### Veo 3 (Google) — Best Atmospheric Landscapes
- **Best for:** Pure landscape/atmospheric shots where cinematic camera motion is the priority
- **Advantage:** Supports audio prompts — add sound design: `ambient river sounds, distant bird calls`
- **Weakness:** Weaker cultural/architectural precision than Kling for traditional subjects

### Seedance 2.0 (ByteDance) — Best Festival / Urban Content
- **Why:** Trained on Douyin (TikTok China) data; 2026 Spring Festival generated massive Chinese cultural feedback loop
- **Best for:** Contemporary urban scenes, festival content, dynamic action with audio sync
- **Resources:** [awesome-seedance-2-prompts on GitHub](https://github.com/YouMind-OpenLab/awesome-seedance-2-prompts) (1,200+ community prompts)

---

## 7. Common Pitfalls

1. **Wrong roof tile color** — Models default to orange/brown; specify "yellow glazed ceramic tiles" for imperial buildings (Forbidden City, Temple of Heaven)
2. **Generic "Asian temple" blending** — Without dynasty and regional qualifiers, models blend Chinese/Japanese/Korean/Thai elements. Always specify: "Qing dynasty," "Naxi timber-frame," "Tibetan rammed-earth"
3. **Western architectural drift** — AI trained on Western data produces Chinese buildings with European proportions. Heavy negative prompting required
4. **Anachronistic intrusions** — Power lines in hutong shots, paved roads in rice terrace aerials, LED lights in lantern festival scenes. Always include a negative prompt list
5. **Badaling default** — Great Wall without section name → congested tourist context. Specify "Jiankou," "Mutianyu," or "Simatai"
6. **Too many simultaneous elements** — 5+ characters + complex architecture + weather + camera movement = degraded output
7. **Open-ended motion** — "Camera moves through the scene" causes hangs. Always specify path, speed, and endpoint
8. **Conflicting lighting** — Don't mix "golden hour" with "overcast soft light"
9. **"Traditional Chinese house" is too vague** — Use: "Qing dynasty siheyuan courtyard, grey brick, moon gate, carved wooden brackets"

---

## 8. Negative Prompt Reference

### Universal (historical/traditional scenes):
```
western architecture, european facades, modern buildings, power lines,
cars, asphalt roads, contemporary signage, foreign branding,
tourist crowds, selfie sticks, watermark, text overlay,
blurry motion artifacts, morphing faces, low quality
```

### Imperial architecture (Forbidden City, temples):
```
japanese architecture, korean architecture, thai temple style,
incorrect roof color, blue tiles, curved western gable,
modern lighting, glass facades
```

### Natural landscapes (Zhangjiajie, Li River, rice terraces):
```
motor boats, roads, power lines, modern structures, fences,
tourists, signage, synthetic materials, artificial lighting
```

### Street / hutong scenes:
```
european facades, neon english signs, western clothing,
modern storefronts, LED strips, contemporary vehicles
```

### Festivals:
```
western holiday decorations, modern LED lights, contemporary clothing,
incorrect lantern color (specify red and gold for Spring Festival)
```

> **Kling note:** In Kling's negative field, write just "X" — not "no X". Extremely long negative prompts make output stiff.

---

## 9. Copy-Paste Prompt Templates

### Template A — Landscape Aerial
```
Aerial drone shot [ascending/descending] over [SPECIFIC LOCATION NAME],
[distinctive geographical features]. [Weather/atmosphere].
Camera [movement path and endpoint].
[Light quality, direction, color temperature].
[Color palette/grade]. [Film style].
No [exclusion list].
```

### Template B — Cultural Architecture
```
[Shot type] of [BUILDING NAME with dynasty/style qualifier,
e.g., "Ming dynasty courtyard siheyuan"],
[specific architectural details: material, color, decorative elements].
[Time of day]. Camera [specific movement].
[Lighting]. [Style].
No western architecture, no modern elements.
```

### Template C — Street Life / Vérité
```
Handheld tracking shot following [subject with specific detail]
through [LOCATION: hutong / tea house / market],
[3 environmental sensory details].
Camera [movement]. [Practical light sources].
35mm grain, documentary vérité style, shallow depth of field.
[Sound design if supported].
```

### Template D — Festival Scene
```
Wide shot of [FESTIVAL: Chinese New Year / Lantern Festival / Dragon Boat],
[specific visual details: "red lanterns strung between buildings /
sky lanterns ascending above river / dragon boats with drummers"].
Camera [movement]. [Crowd energy]. [Lighting: lantern glow / firecracker haze].
Warm festive palette. Cinematic grade.
No [anachronistic elements].
```

### Template E — Misty Mountain / Ink Painting Style
```
Wide cinematic shot of [MOUNTAIN LOCATION] at dawn,
[mist behavior: "mist rolling through valleys / cloud sea below summit ridge"].
Camera [very slow movement].
Monochromatic grey-green palette, overcast soft light,
ink wash painting aesthetic. Meditative, still mood.
National Geographic style with Chinese ink painting influence.
```

---

## 10. Recommended Workflow (2026)

| Goal | Best Tool | Key Technique |
|---|---|---|
| Traditional architecture, hutongs, festivals | **Kling 3.0** | Native cultural training data |
| Specific regional architecture (Naxi, Tibetan) | **Wan 2.2** | Chinese-language prompts |
| Have reference photos of the location | **Runway Gen-4** | Image-to-video pipeline |
| Festival content, urban scenes with audio | **Seedance 2.0** | Douyin-trained, strong audio sync |
| Pure landscape / atmosphere | **Veo 3** | Add audio prompts for immersion |
| Contemporary urban, narrative sequences | **Sora 2** | Shot Stack temporal structure |

---

## References

- [Kling AI Prompts Complete Guide — VEED](https://www.veed.io/learn/kling-ai-prompting-guide)
- [Kling 2.6 Pro Prompt Guide — fal.ai](https://fal.ai/learn/devs/kling-2-6-pro-prompt-guide)
- [Kling 3.0 Prompting Guide — Atlabs AI](https://www.atlabs.ai/blog/kling-3-0-prompting-guide-master-ai-video-generation)
- [Veo on Vertex AI Video Prompt Guide — Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide)
- [Runway Camera Terms, Prompts & Examples](https://help.runwayml.com/hc/en-us/articles/47313504791059-Camera-Terms-Prompts-Examples)
- [Seedance, Kling and the Chinese AI Video Ecosystem — ChinaTalk](https://www.chinatalk.media/p/seedance-kling-and-the-chinese-ai)
- [Kling AI: How China's Precision Strategy Is Reshaping Global Video Generation](https://chinacompany.substack.com/p/kling-ai-how-chinas-precision-strategy)
- [Negative Prompts for Kling, Veo, and Wan — Artlist](https://artlist.io/blog/negative-prompts-ai-video/)
- [Awesome Seedance 2.0 Prompts — GitHub](https://github.com/YouMind-OpenLab/awesome-seedance-2-prompts)
- [Wan 2.1 Guide with Practical Examples — DataCamp](https://www.datacamp.com/tutorial/wan-2-1)
- [Master Cinematic AI Video Prompts 2026 — TrueFan](https://www.truefan.ai/blogs/cinematic-ai-video-prompts-2026)
- [7 Best AI Drone Video Shot Prompts — HitPaw](https://edimakor.hitpaw.com/video-editing-tips/drone-shot-ai-prompt.html)
- [China's new AI video tools close the uncanny valley — Fast Company](https://www.fastcompany.com/91489530/chinas-new-ai-video-tools-close-the-uncanny-valley-for-good)
