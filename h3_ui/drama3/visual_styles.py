"""Sound Stage rendering presets and compatibility with saved art direction."""

import re


# Each preset: (use-case group, name, what it's for, preview scene, rendering text). The text describes
# rendering, not story content: it applies to portraits and every scene. The preview scene shows the
# style on a subject typical of its use, with faces, hands and a finished setting to judge it by.
_PRESETS = (
    ("Drama & fiction", "Digital painting",
     "Any genre for teens and adults; the all-round default",
     "A medium-wide shot at dusk: a weathered sea captain in a wool coat grips the wheel of a fishing "
     "boat while a young deckhand points past him at a lighthouse beam cutting through the storm; rain "
     "streaks the wheelhouse windows and waves break over the bow.",
     "Full-color narrative digital painting with believable anatomy, distinct faces and natural "
     "expressions. Solid forms and materials, restrained visible brushwork, clear "
     "detail on faces and hands, and coherent local colors. Keep gestures readable."),
    ("Drama & fiction", "Realistic photo",
     "Grounded contemporary drama, crime and documentary reenactment",
     "A medium shot in a small-town diner at night: a tired detective in her forties sits in a red vinyl "
     "booth, coffee in both hands, studying a spread of photographs; neon from the window falls across "
     "the chrome counter where a waitress refills a coffee pot.",
     "Full-color live-action film still with natural human proportions, individual facial "
     "features, subtle skin texture and believable fabric and surfaces. Physically "
     "plausible light, restrained color grading and fine film grain. Avoid fashion-ad "
     "posing and beauty retouching."),
    ("Drama & fiction", "American comic",
     "Action, thriller and mystery with bold, punchy staging",
     "A dynamic low-angle shot on a rain-soaked rooftop: a courier in a torn jacket leaps the gap between "
     "two brick buildings, a satchel clutched to her chest, water towers and fire escapes behind her and "
     "a searchlight sweeping the alley below.",
     "American comic-book illustration with bold brush-ink contours, varied line "
     "weight, sculpted spot blacks and selective cross-hatching. Clearly modeled "
     "anatomy, distinct faces and expressive gestures; rich flat colors with crisp "
     "shadow shapes and restrained halftone texture. Keep the story's clothing and "
     "physique without adding superhero costumes or exaggerated muscles. One continuous "
     "image, no panel borders, speech balloons, captions or lettering."),
    ("Drama & fiction", "Oil painting",
     "Historical, period and literary stories",
     "A candlelit study: an elderly scholar reads a letter at a heavy oak desk crowded with books, a quill "
     "and an hourglass, while his granddaughter waits in the doorway holding a lantern; tall windows show "
     "a snowy courtyard.",
     "Full-color oil painting with visible bristle strokes, layered opaque pigments "
     "and subtle canvas texture. Model faces, hands and objects with believable "
     "proportions and clear light and shadow, and natural material colors. Show the "
     "painting itself, without a picture frame, gallery wall or added antique costumes."),
    ("Drama & fiction", "Chinese ink painting",
     "Wuxia, xianxia, Chinese myth and history",
     "A wide shot of a mountain pass at dawn: a lone swordswoman in layered robes stands on a stone bridge "
     "above a waterfall, her sleeves lifted by the wind, pines clinging to the cliffs and a temple roof "
     "rising from the valley below.",
     "Full-color traditional Chinese ink-and-color painting on fine rice paper. Confident brush contours "
     "with varied pressure, finely drawn faces and hands, layered mineral colors in transparent washes and "
     "ink gradations that model rock, water, foliage and fabric. Distinct figures with believable "
     "proportions and readable expressions. Keep the story's own era and clothing; no seals, calligraphy, "
     "scroll borders or blank paper standing in for the setting."),
    ("Drama & fiction", "Watercolor",
     "Gentle slice of life, memoir and poetry",
     "A quiet morning kitchen: a grandmother and a boy share breakfast at a table by an open window, steam "
     "rising from their bowls, potted herbs on the sill and a bicycle leaning against the garden fence "
     "outside.",
     "Full-color watercolor illustration with transparent layered washes, soft pigment "
     "edges and selective fine pencil contours. Preserve facial features, hands and "
     "important objects with controlled edges. Subtle paper grain, believable proportions "
     "and consistent pigment density; avoid splatter or unfinished white gaps across "
     "focal subjects."),
    ("Anime, manga & games", "Anime",
     "Youth, romance, fantasy and light-novel stories",
     "A medium shot on a train platform at sunset: one classmate holds out a folded letter while the "
     "other hesitates, school bags over their shoulders, a commuter train pulling in and power lines "
     "against an orange sky.",
     "Full-color 2D dramatic animation still with controlled fine outlines, consistent character proportions "
     "and clean two- or three-tone cel shading. Individually designed faces with restrained, "
     "readable expressions. Avoid chibi proportions, exaggerated reaction symbols and decorative bloom."),
    ("Anime, manga & games", "Japanese comic",
     "Dramatic manga-style action and character stories",
     "A festival night: a young swordsman kneels to mend a girl's broken sandal strap on a crowded shrine "
     "path, paper lanterns glowing overhead, food stalls on both sides and fireworks opening above the "
     "torii gate.",
     "Full-color Japanese manga illustration with precise pen lines, delicate "
     "facial detail and controlled variation in line weight. Rich local colors, "
     "clean cel-shaded shadows and selective color gradients. Consistent stylized "
     "proportions, distinct character designs and expressive but believable acting, "
     "with legible eyes and gestures. One continuous image, "
     "no panels, speech balloons, lettering or decorative reaction symbols."),
    ("Anime, manga & games", "Pixel art",
     "Game lore, retro sci-fi and gaming podcasts",
     "A side view of a torchlit dungeon: an adventurer with a backpack and lantern faces a small dragon "
     "curled on a pile of gold, stone pillars, a locked iron gate and glowing mushrooms around them.",
     "Full-color narrative pixel art with a consistent coarse pixel grid, deliberate pixel clusters "
     "and a limited shared palette. Crisp stepped edges without smoothing or mixed pixel "
     "sizes; selective dithering for shading. Simplify forms into readable silhouettes "
     "while preserving each character's hair, outfit and key facial cues; no game "
     "interface or decorative border."),
    ("Kids & family", "Children's book",
     "Bedtime and picture-book stories for young children",
     "A cozy bedtime scene: a little girl in pajamas reads a picture book to her stuffed rabbit and a "
     "sleepy cat inside a blanket fort, fairy lights strung across the bedroom wall and a full moon in "
     "the window.",
     "Full-color hand-painted gouache and colored-pencil storybook illustration. Matte opaque "
     "color shapes, subtle pencil edges and gentle paper texture; simplified but "
     "consistent anatomy, distinct character silhouettes and legible expressions. "
     "Keep tension or sadness when the story has it instead of making everything cheerful."),
    ("Kids & family", "2D cartoon",
     "Comedy and lively stories for kids",
     "A backyard mishap: a boy and his overexcited dog are tangled in a garden hose spraying water "
     "everywhere while his older sister holds a soaked birthday cake with a deadpan look; a picket fence "
     "and a treehouse behind them.",
     "Full-color 2D cartoon illustration with clean outlines, simple rounded shapes, "
     "bold flat colors and minimal cel shading. Consistent character proportions, distinct "
     "silhouettes and expressive faces, serious when the story is. One continuous image "
     "without speech bubbles, lettering or comic panels."),
    ("Kids & family", "3D animation",
     "Family adventure with feature-animation appeal",
     "A wide adventure shot: a young inventor and her small round robot ride a homemade glider over a "
     "seaside village of colorful rooftops, laundry lines and a windmill, the inventor laughing while "
     "the robot holds on tight.",
     "Full-color stylized 3D animated feature still with appealing, consistent character proportions "
     "and expressive but restrained faces. Characters, props and environments share one coherent "
     "3D production design: rounded modeled forms, clean geometry, clear silhouettes and restrained "
     "surface texture. Build stone, wood and cloth from readable larger shapes with subtle grain "
     "and selective wear. Soft bounced light, smooth tonal transitions and controlled specular "
     "highlights distinguish materials; wet surfaces have broad reflections rather than dense "
     "sparkling edges. Keep backgrounds complete with quieter detail and atmospheric separation "
     "so the action reads first. Avoid crunchy microtexture, etched outlines, sharpening halos "
     "and harsh local contrast. Keep eyes and skin natural "
     "within the stylization, without a glossy plastic finish."),
    ("Explainers & slides", "Flat illustration",
     "Training, onboarding and how-to explainers",
     "A team onboarding moment: a mentor points at simple icons on a laptop screen while three new "
     "colleagues lean in around a desk with a plant, coffee mugs and a blank whiteboard.",
     "Full-color flat business illustration with clean geometric shapes, solid "
     "color fills and minimal shading. Use simplified people and recognizable "
     "objects to explain one idea at a time, with consistent proportions and a "
     "small coordinated color palette. Clear visual hierarchy, generous spacing "
     "and a plain light background; leave an uncluttered area for slide text "
     "to be added later. Avoid decorative clutter, lettering, logos and invented data."),
    ("Explainers & slides", "Isometric illustration",
     "Systems, processes and product walkthroughs",
     "How an online order travels: a stocked warehouse shelf, a packing table with boxes, a delivery van "
     "and a doorstep with a parcel, linked by a clear path on one compact platform.",
     "Full-color isometric business illustration with parallel projection, "
     "consistent object scale and clean geometric volumes. Use restrained "
     "shading and coordinated colors to distinguish components. Show the "
     "requested system, process or workspace through clearly arranged objects "
     "and meaningful spatial relationships. Keep one coherent scene on a "
     "simple light background, with generous margins for slide text added "
     "later. No labels, logos, invented metrics or decorative dashboards."),
    ("Explainers & slides", "Minimal vector",
     "Keynotes and one-idea concept slides",
     "A product launch as one idea: a single rocket lifting off from an open laptop, a few small stars "
     "around it.",
     "Full-color minimal vector-style illustration with crisp edges, simple shapes "
     "and consistent stroke widths. Reduce the requested idea to a few recognizable "
     "elements with strong visual hierarchy, using a small coordinated palette "
     "on a plain light background. Keep ample negative space for slide text added "
     "later and maintain clarity at presentation size. No texture, ornamental "
     "detail, lettering, logos or invented charts and numbers."),
    ("Explainers & slides", "Editorial illustration",
     "Commentary, news analysis and idea podcasts",
     "Information overload as a metaphor: a person reads calmly at a small desk while a towering wave of "
     "envelopes and notification bubbles curls above them.",
     "Full-color editorial business illustration built around one clear "
     "visual metaphor for the requested idea. Use simplified recognizable "
     "forms, confident silhouettes, restrained texture and a coordinated "
     "palette. Make the relationship between subjects easy to understand "
     "without labels; keep the focal concept distinct from a simple "
     "background and leave breathing room for slide text added later. "
     "Avoid unrelated symbols, stock handshake imagery, lettering, logos "
     "and invented data."),
)
# Medium-specific finish keeps the same visual language across figures and scenery. Keep the
# original wording separately so saved art direction remains recognizable after these refinements.
_PRESET_FINISH = {
    "Digital painting": (
        " Group light and shadow into clear masses, with selective crisp edges at the focal action "
        "and softer transitions elsewhere. Carry the same controlled brushwork through scenery; "
        "suggest distant texture with larger color shapes. Keep highlights selective and material "
        "grain subtle, without gritty overlays or sharpening halos."),
    "Realistic photo": (
        " Use a consistent light direction, natural tonal rolloff and restrained reflections. "
        "Keep skin, fabric and background surfaces at a believable scale of detail; fine grain "
        "stays faint in smooth areas. Let contrast and focus fall away naturally with distance. "
        "Avoid HDR-like local contrast, outlined textures and sharpening halos."),
    "American comic": (
        " Organize shadows into readable connected shapes. Use heavier contours for focal forms "
        "and lighter, simpler marks for distant architecture. Confine cross-hatching and halftone "
        "to selected shadow areas, leaving clean color passages between them. Maintain coherent "
        "perspective and clear object boundaries without covering every surface in ink texture."),
    "Oil painting": (
        " Let brushstrokes follow the planes of faces, fabric and architecture, with broader, "
        "quieter passages away from the focal action. Unify the scene with consistent light and "
        "grouped tonal masses. Keep canvas grain faint and impasto selective; distinguish materials "
        "through color and edge handling rather than a uniform crust of rough paint."),
    "Chinese ink painting": (
        " Balance decisive structural strokes with broad quiet washes; use dry-brush texture "
        "selectively on nearby rock or bark. Separate distant forms with lighter ink and simpler "
        "contours while keeping architecture legible. Leave breathing room within the depicted "
        "setting, with subtle paper grain and controlled pigment pooling."),
    "Watercolor": (
        " Build objects and scenery from clear transparent color shapes, with selective hard edges "
        "at important boundaries and soft or lost edges elsewhere. Keep washes luminous and local "
        "colors distinct. Reserve granulation and pencil marks for small accents; background forms "
        "stay recognizable through shape and value rather than dense stippling or muddy pigment."),
    "Anime": (
        " Pair clean character shading with a coherent painted background: stable perspective, "
        "readable architectural planes and simplified natural forms. Group foliage and surface "
        "detail into larger color masses. Keep outlines controlled, highlights selective and "
        "distant contrast reduced, without noisy photo textures or hard rims on every object."),
    "Japanese comic": (
        " Use fine, deliberate contours and readable shadow shapes throughout the setting. "
        "Simplify distant structures and vary line density to emphasize the action. Keep hatching "
        "and screentone sparse and localized, with clean color passages and clear perspective; "
        "avoid dense scratchy marks over skin, fabric and background alike."),
    "Pixel art": (
        " Construct scenery on the same pixel grid as the figures, with connected clusters for "
        "walls, foliage and shadows. Separate near and far planes through palette and cluster "
        "size, keeping pixel edges crisp at every depth instead of using lens blur. Limit "
        "dithering to selected transitions and avoid scattered single-pixel noise."),
    "Children's book": (
        " Build figures and scenery from the same rounded, clearly painted shapes. Group plants, "
        "furniture and buildings into simple readable forms with a few descriptive accents. "
        "Use gentle value changes and selective pencil marks, keeping paper grain faint and "
        "avoiding scratchy coverage or muddy color buildup."),
    "2D cartoon": (
        " Use consistent contour weight for nearby forms and simpler contours in the distance. "
        "Give scenery the same flat fills and limited shadow tones as the figures, with clean "
        "joins and clear overlaps. Describe materials through shape and color; keep gradients, "
        "surface grain and glossy highlights minimal."),
    "3D animation": (
        " Maintain coherent scale and perspective across the set, with contact shadows grounding "
        "figures and props. Keep fine surface detail sparse even on nearby objects."),
    "Flat illustration": (
        " Keep edges, stroke weights and corner shapes consistent across people and objects. "
        "Separate overlapping forms with clear color boundaries and generous spacing; use only "
        "a few functional shadow shapes. Keep all surfaces flat and clean without grain, bevels "
        "or photorealistic lighting."),
    "Isometric illustration": (
        " Align every object to the same isometric axes, with parallel edges and coherent joins. "
        "Use a consistent light direction and a small set of flat shades per material. Keep "
        "small parts simplified and distinct, without photographic textures, lens blur or "
        "perspective convergence."),
    "Minimal vector": (
        " Make every contour deliberate, with smooth curves, aligned joins and consistent "
        "spacing. Keep fills flat and silhouettes distinct, with no stray marks, tiny surface "
        "details, lens blur or simulated paper grain."),
    "Editorial illustration": (
        " Give every metaphorical element a clear silhouette and intentional overlap. Use broad "
        "color shapes and a consistent edge language, concentrating contrast on the relationship "
        "that explains the idea. Keep any print texture faint and uniform, without obscuring "
        "boundaries or adding incidental background detail."),
}
_ORIGINAL_PRESETS = {name: text for _group, name, _use, _preview, text in _PRESETS}
_BARE_PRESETS = {name: text + _PRESET_FINISH[name] for name, text in _ORIGINAL_PRESETS.items()}
# Finish concerns coherent forms, not equal sharpness throughout the frame. The slide presets keep
# their deliberately plain backgrounds.
SETTING_DETAIL = (" Render a complete, coherent setting with legible architecture, objects and landscape. "
                  "Keep background detail subordinate to the focal action, with depth of field appropriate "
                  "to the shot; avoid smeared geometry and unfinished areas.")
SLIDE_STYLES = tuple(name for group, name, *_ in _PRESETS if group == "Explainers & slides")
# Projects saved with earlier preset text keep it as their Art direction (and their shot plan is keyed
# by it), so those texts still count as the preset rather than as Custom.
_EARLIER_DETAIL = (" Paint the setting with the same finish and focus as the figures: legible architecture, "
                   "objects and landscape, never a blurred, smeared or sketchy backdrop.")
_PREVIOUS_DETAIL = _EARLIER_DETAIL.replace(" Paint", " Render", 1)
_LEGACY_PRESETS = {text: name for presets in (_ORIGINAL_PRESETS, _BARE_PRESETS)
                   for name, text in presets.items()}
_LEGACY_PRESETS[("Full-color stylized 3D animated feature still with appealing, consistent character proportions "
                 "and expressive but restrained faces. Rounded modeled forms, tactile cloth, "
                 "convincing material differences and clear silhouettes. Keep eyes and skin natural "
                 "within the stylization, without a glossy plastic finish.")] = "3D animation"
for _presets in (_ORIGINAL_PRESETS, _BARE_PRESETS):
    _LEGACY_PRESETS[_presets["Digital painting"].replace(
        "clear detail on faces and hands", "sharper detail around faces and hands")] = "Digital painting"
    _LEGACY_PRESETS[_presets["Japanese comic"].replace(
        "with legible eyes and gestures. One", "with legible eyes and gestures. Full color throughout. One")] = \
        "Japanese comic"
_LEGACY_PRESETS.update({text + detail: name for text, name in list(_LEGACY_PRESETS.items())
                        for detail in (SETTING_DETAIL, _EARLIER_DETAIL, _PREVIOUS_DETAIL)
                        if name not in SLIDE_STYLES})
STYLE_PRESETS = {name: text if name in SLIDE_STYLES else text + SETTING_DETAIL
                 for name, text in _BARE_PRESETS.items()}
STYLE_GROUPS = [{"name": group, "presets": [name for g, name, *_ in _PRESETS if g == group]}
                for group in dict.fromkeys(group for group, *_ in _PRESETS)]
STYLE_USES = {name: use for _group, name, use, _preview, _text in _PRESETS}
STYLE_PREVIEWS = {name: preview for _group, name, _use, preview, _text in _PRESETS}
DEFAULT_STYLE = "Digital painting"
CUSTOM_STYLE = "Custom"


def preset_for(text):
    """The preset an Art direction text matches, or Custom once it's been edited."""
    text = (text or "").strip()
    return next((name for name, value in STYLE_PRESETS.items() if value == text), None) \
        or _LEGACY_PRESETS.get(text, CUSTOM_STYLE)


def reference_direction(style):
    """The Art direction for portraits: without SETTING_DETAIL, which asks for scenery behind a
    portrait meant to stand on a plain background."""
    style = style or ""
    for detail in (SETTING_DETAIL, _EARLIER_DETAIL, _PREVIOUS_DETAIL):
        style = style.replace(detail.strip(), "")
    return re.sub(r"[ \t]{2,}", " ", style).strip()
