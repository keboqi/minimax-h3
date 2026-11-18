"""Illustration beats for the three built-in Soundstage scripts, on their dialogue cues."""

BLACKTHORN = {
    "characters": {
        "elena": "Elena, a woman in her early thirties and a marine investigator, narrow face, dark hair in a low ponytail, lean build, navy storm parka with a brass compass clipped to the chest",
        "marcus": "Marcus, a man in his mid-sixties and a former lighthouse keeper, broad shoulders, weathered face, close grey beard, olive oilskin coat",
    },
    "locations": {
        "reef": "granite lighthouse on black sea stacks at night, iron gallery and diamond-paned lantern crown",
        "watchroom": "round granite watchroom, rusted iron door, dark timber floor, porthole and wall shelf",
        "gallery": "slick exterior walkway with a low granite parapet and iron railing high above the reef",
        "chamber": "upper lantern room with diamond-paned windows, iron stairs and wet flagstones",
        "vault": "drowned stone vault under the reef, barnacled arches and massive lead bulkheads",
    },
    "props": {
        "lantern": "handheld brass storm lantern with a heavy glass globe and amber flame",
        "console": "dark green maritime console with exposed copper coils and an unlettered brass voltmeter",
        "radio": "black marine transceiver with a coiled handset and a single amber indicator",
        "core": "lighthouse beacon with rings of faceted prisms around a sphere in a brass cage",
    },
}
BLACKTHORN_SHOTS = (
    (0, "cut", "The lighthouse stands dark above the breaking reef.",
     "Unoccupied wide view of @reef(unlit crown), black rocks breaking the slate-blue sea. Hold the tower silhouette clear against layered storm clouds.", "rain", "push_in"),
    (1, "cut", "Elena wrestles the watchroom door shut.",
     "Eye-level medium shot of @elena(both hands on the heavy door) inside @watchroom, a narrow strip of cold exterior light at the threshold."),
    (2, "cut", "Marcus brings his lantern to the console.",
     "Low medium view of @marcus(setting @lantern beside @console) in @watchroom, an amber pool on the wet boards.", "flicker"),
    (3, "cut", "The voltmeter is still live.",
     "Object-only insert of @console(needle pinned high) in @watchroom, the curved glass reflecting @lantern. No people, hands or human reflections.", "push_in"),
    (7, "cut", "Elena listens to the casing.",
     "Floor-level medium close-up in @watchroom. @elena(kneeling, one ear close to @console) watches a ring of droplets tremble on the casing. She is the only person in frame.", "subtle_sway"),
    (8, "cut", "The transceiver breaks the silence.",
     "Object-only close-up of @radio(indicator illuminated) on the shelf of @watchroom, coiled cable trailing toward the edge. No people or hands.", "vignette"),
    (11, "cut", "Vance's warning interrupts the radio exchange.",
     "Two people in a medium shot in @watchroom. @elena(left of center, handset held to her ear) looks sharply toward @marcus(right of center, torso turned toward the stair). The cable leads back to @radio on the shelf.", "cold_tint", "subtle_sway"),
    (12, "cut", "They step onto the exposed gallery.",
     "Wide low angle of two figures emerging onto @gallery. @elena(in front, reaching for the iron rail) braces against the gale. @marcus(one step behind, holding @lantern low) shields the flame with his free hand.", "rain", "pan_right", "cold_tint"),
    (14, "cut", "The waves stand in a ring.",
     "Unoccupied high view from @gallery toward the black water, a continuous rim of white foam around the eerily smooth center.", "pull_out"),
    (16, "cut", "Elena and Marcus look up at the lantern crown.",
     "From rail height at @gallery, frame only @elena(nearer the camera, chin raised) and @marcus(just beyond her, looking up past the iron bars). The crown of @reef looms above them.", "rain", "lightning", "pan_up"),
    (18, "cut", "A prism shatters on the lantern room floor.",
     "Object-only floor-level close-up in @chamber of a broken prism at the foot of @core, its brass rings rising into darkness. No people or human reflections.", "impact_shake"),
    (19, "cut", "The voice of Aegis recognizes the keeper.",
     "Medium two-shot in @chamber. @marcus(near @core, staring into its rings in disbelief) is the focus. @elena(across from him, looking at his face) remains clearly visible in the cold window light."),
    (23, "dissolve", "Aegis opens the receiver to the flooded vault.",
     "Unoccupied view through @vault with silvery bubbles below the arches and a faint cyan glow escaping the bulkhead seam.", "fog", "push_in", "cold_tint"),
    (26, "dissolve", "Elena listens for the voice below the rock.",
     "Solo close profile of @elena(listening intently beside @core) in @chamber, teal light on her cheek and soft iron window tracery behind her.", "dream_bloom"),
    (28, "cut", "The beacon ignites.",
     "Low wide view inside @chamber as @core floods the room with white-gold light. Below it, @elena(left of center, forearm shielding her eyes) and @marcus(right of center, head angled down against the glare) have separate silhouettes and readable coats.", "flash", "pull_out"),
    (29, "cut", "The beam crosses the storm-dark sea.",
     "Unoccupied wide exterior of @reef(beacon lit), a clean beam out over the swells, white foam giving way to the quiet horizon.", "pull_out"),
)

KUNLUN = {
    "characters": {
        "xinghe": (
            "沈星河, a Chinese woman in her late twenties and an expedition researcher, narrow face, "
            "short black hair tucked behind one ear, lean build, rust-red insulated jacket and grey gloves"
        ),
        "yuchi": (
            "尉迟烈, a Chinese man in his late forties and an expedition captain, broad weathered face "
            "with lines around the eyes, cropped black hair greying at the temples, stocky build, "
            "dark indigo expedition coat with a fur-lined hood"
        ),
    },
    "locations": {
        "ridge": "high Kunlun ridge, blue-black crags and wind-scoured snow above a deep ice gorge",
        "outpost": "stone mountain outpost with an ancient bronze gate, frost on the floor and a broad stone workbench",
        "walkway": "narrow ice-covered cliff walkway with an iron chain above the gorge",
        "hall": "vaulted underground bronze observatory, large dark floor tiles and tall engraved bronze columns",
        "ice_river": "slow luminous river beneath translucent ice, cavern walls fading to dark cobalt",
    },
    "props": {
        "lamp": "portable expedition searchlight with a square warm lens and worn yellow housing",
        "dial": "circular stone star dial with bronze rings, a single pointer and a carved star pattern",
        "radio": "shortwave expedition radio with a rubber aerial, coiled handset and small status light",
        "armillary": "room-sized ancient armillary sphere of nested engraved bronze rings enclosing the Tian Shu core",
        "mirror": "round polished bronze basin filled with dark water that catches cyan reflections",
    },
}
KUNLUN_SHOTS = (
    (0, "cut", "风雪封住昆仑前哨。",
     "Unoccupied wide view of @ridge with one small outpost light in the blue twilight. Reserve the foreground for sculpted rock and snow ridges.", "snow", "push_in"),
    (1, "cut", "沈星河合上青铜闸门。",
     "Solo medium shot inside @outpost. @xinghe(in profile, shoulder and both palms braced against the bronze gate) forces it closed. Cold light from the narrowing gap outlines her face and hands."),
    (2, "cut", "探照灯照亮冰霜和石盘。",
     "Solo low medium shot inside @outpost. @yuchi(setting @lamp beside @dial) leans over the stone workbench. Warm light grazes his face, the fur trim at his shoulders and the frost on the bench.", "warm_glow", "dust"),
    (3, "cut", "沈星河指给尉迟烈看对准北极星的指针。",
     "Medium two-shot across the stone workbench inside @outpost. "
     "@xinghe(left of center, pointing to the markings on @dial) addresses "
     "@yuchi(right of center, looking down at the pointer with his hands lowered). "
     "The pointer aligns with the outer star carving between them. Light from @lamp keeps both faces "
     "and coats readable against the closed bronze gate.", "push_in"),
    (7, "cut", "底座传来微弱的呼吸。",
     "Solo ground-level close view inside @outpost. @xinghe(crouched in profile with one ear near the base of @dial) listens as grains of frost shift in the seam beneath it.", "vignette", "subtle_sway"),
    (8, "cut", "短波电台突然响起。",
     "Object-only close-up of @radio(status light on) on the stone workbench of @outpost, edge-lit by @lamp. No people or hands."),
    (11, "cut", "撤退的警告让两人停住。",
     "Two people in a medium shot inside @outpost. @xinghe(left of center, holding the handset from @radio to her ear) freezes at the warning. @yuchi(right of center, turning toward the bronze gate) glances over his shoulder. The cold gate fills the space behind them."),
    (12, "cut", "两人踏上悬崖栈道。",
     "Wide shot of two people on @walkway. @yuchi(one step ahead, braced with one hand on the chain) checks the ice underfoot. @xinghe(behind him, reaching for the next link) follows. Keep their coat colors readable against the snow and leave most of the lower frame to the drop.", "snow", "pan_left", "cold_tint"),
    (14, "cut", "深渊里的暗河亮起来。",
     "Unoccupied steep overhead view from @walkway toward @ice_river, a narrow ribbon of blue light far below fractured ice shelves.", "pan_down", "cold_tint"),
    (16, "cut", "栈道上方现出浑天仪。",
     "Low upward angle from @walkway. @xinghe(nearer the camera, pointing up) draws the gaze toward @armillary in the high cliff opening. @yuchi(beside her, one hand still on the chain) follows her gaze. Keep their faces in separate profiles.", "fog", "pan_up"),
    (18, "cut", "玉圭在青铜大殿内摔碎。",
     "Object-only floor-level close-up of pale jade fragments across the dark tiles of @hall, @armillary reflected faintly in the floor. No people, hands or human reflections.", "impact_shake"),
    (19, "cut", "天枢回应，两人神色一变。",
     "Medium two-shot beside @armillary in @hall. @yuchi(right of center, staring at the core in disbelief) catches light from the bronze rings across his face. @xinghe(left of center, studying his reaction) catches a softer reflection. Keep both faces and distinct coats readable."),
    (23, "dissolve", "水镜接通寒渊的回波。",
     "Object-only close-up of @mirror on the floor of @hall, cyan ripples spreading from a single point. No people, hands or human reflections.", "push_in", "dream_bloom"),
    (25, "dissolve", "冰河深处传来守墟古灵的声音。",
     "View under the translucent roof of @ice_river with a small star-like light beyond the layered ice, no figure visible.", "fog", "dream_bloom"),
    (28, "cut", "天枢点亮主烽燧。",
     "Symmetrical wide interior of @hall with @armillary aligned around a bright cyan center. Below the rings, @xinghe(left of center, head raised toward the core) and @yuchi(right of center, forearm raised against the glare) stand apart, their coat colors visible in the reflected light.", "flash", "push_in"),
    (29, "cut", "苍青光柱穿出昆仑风雪。",
     "Unoccupied distant view of @ridge with a narrow shaft of cyan light rising from the summit through windblown snow.", "snow", "pull_out", "cold_tint"),
)

RAIN_ALLEY = {
    "characters": {
        "achuan": "阿川, a Chinese man in his early twenties, lanky with a long face and tousled black hair, oversized faded green rain jacket",
        "zhixia": "许知夏, a Chinese woman in her late twenties and a clockmaker's apprentice, slim build, oval face and chin-length black bob, navy work apron over burgundy knitwear",
        "chen": "陈伯, a Chinese man in his seventies and a watchmaker, slightly stooped, lined cheeks, silver hair and wire-rimmed glasses, brown cardigan with rolled cuffs",
        "lin": "林姨, a Chinese woman in her late sixties, upright posture, softly rounded face, grey hair pinned into a low bun, plum raincoat",
    },
    "locations": {
        "street": "narrow old neighborhood lane, tiled eaves, wet paving and soft yellow shop lights at dusk",
        "shop": "small clock repair shop, green awning, oak counter, clocks on the rear wall and a low street-facing sill",
    },
    "props": {
        "radio": "old cream portable radio with a black dial, brass speaker grille and loose battery cover",
        "note": "small timetable note folded several times, ink marks too small to read",
        "mic": "round silver tabletop broadcast microphone with a worn cloth lead",
        "clock": "small walnut mantel clock with a brass pendulum behind beveled glass",
        "basket": "woven shopping basket with a curved wicker handle",
        "tickets": "two pale paper train tickets, creased at the corners with no legible lettering",
    },
}
RAIN_ALLEY_SHOTS = (
    (0, "cut", "小雨落在老街的屋檐下。",
     "Wide blue-hour view of @street with one warm window at @shop. Wet paving reflects the window along the empty lane.", "rain", "push_in"),
    (1, "cut", "阿川抱着收音机钻进雨棚。",
     "Solo medium shot at awning height. @achuan(ducking under the awning with @radio hugged to his chest) reaches the dry patch outside @shop. Keep the rain behind him in the street.", "rain", "pan_right"),
    (3, "cut", "知夏指给阿川看空着的电池槽。",
     "Medium two-shot across the counter of @shop. @zhixia(left of center, pointing to the loose battery cover) leans over @radio as @achuan(right of center, cradling the radio and giving her a sheepish grin) meets her gaze."),
    (7, "cut", "陈伯攥着折旧的时刻表。",
     "Three people at the oak counter in @shop. @chen(behind the counter, gripping @note) looks down at the folds. In front of him, @zhixia(holding out a battery) glances back at him as @achuan(reaching for the battery) sets his radio down.", "push_in"),
    (11, "cut", "打烊后，陈伯把纸摊平。",
     "Solo medium close-up at the oak counter of @shop. @chen(unfolding @note with both hands) looks down through his glasses. Frame his face and hands together, with a blurred closed sign beyond.", "warm_glow"),
    (13, "cut", "那只钟早就修好了。",
     "Object-only insert of @clock on a shelf in @shop, its brass pendulum still and the wall clocks softly out of focus. No people, hands or human reflections.", "dust"),
    (16, "cut", "陈伯说起那些拖延的日子。",
     "Eye-level two-shot in @shop with @chen(in profile, eyes lowered) beside the shelf holding @clock. @zhixia(across the counter, listening with her hands still) watches his face. The work lamp provides the only warm light.", "push_in", "dream_bloom"),
    (20, "cut", "知夏提醒师父，林姨一直懂他。",
     "Reverse medium two-shot in @shop. @zhixia(in focus with both palms flat on the counter) meets the eyes of @chen(nearer the camera in three-quarter profile). The damp blue window stays softly visible."),
    (24, "cut", "阿川放下收音机，知夏把话筒递给陈伯。",
     "Medium three-shot in @shop. @achuan(by the window, setting @radio on the sill) is just behind the counter. @zhixia(at the left of the counter, sliding @mic forward) offers it to @chen(on the right, reaching for its base). Leave space between all three faces."),
    (27, "dissolve", "七点整，收音机亮了。",
     "Object-only close-up of @radio glowing inside the window of @shop, rain streaking the glass behind it and @mic out of focus in the foreground. No people, hands or human reflections.", "warm_glow", "push_in", "rain"),
    (28, "cut", "陈伯开口，请林姨留一个座位。",
     "Solo close profile of @chen(speaking into @mic, eyes fixed on the open doorway of @shop), the quiet clock faces behind him.", "push_in"),
    (32, "cut", "门外传来篮子碰门槛的轻响。",
     "Floor-level detail across the threshold of @shop. The base of @basket touches the stone step with the wet lane beyond. Frame only the lower half of the basket and the threshold. No people, hands or human reflections.", "subtle_sway"),
    (33, "cut", "林姨走进灯下，陈伯转过身。",
     "Medium two-shot in @shop. @lin(in the doorway, holding @basket by her side) calls into the room. @chen(behind the counter, turning to face her) lifts his eyes from @mic. Warm light reaches both faces.", "warm_glow"),
    (35, "cut", "林姨拿出早已买好的两张车票。",
     "Medium close two-shot in @shop with hands and faces in frame. @lin(on the doorway side of the counter, sliding @tickets into an open palm) watches @chen(on the shop side, taking the tickets and looking up at her). The worn ticket corners are visible between them.", "film_grain"),
    (40, "cut", "陈伯轻轻推动钟摆。",
     "Medium close two-shot at the oak counter of @shop. @chen(leaning down to nudge the brass pendulum of @clock) looks through his glasses at the mechanism. @lin(on the other side of the clock, smiling faintly) watches his hand. Both faces are directly visible.", "push_in"),
    (42, "dissolve", "两人准备一起出门。",
     "Four people seen from @street just outside @shop. @chen(at the threshold, carrying @clock) stands beside @lin(holding @basket, turning to leave with him). In the warm room behind them, @zhixia(near the counter, watching them go) and @achuan(by the window, raising a farewell hand) remain visible.", "rain", "pull_out", "warm_glow"),
)

EXAMPLE_VISUALS = {
    "en": (BLACKTHORN, BLACKTHORN_SHOTS),
    "zh": (KUNLUN, KUNLUN_SHOTS),
    "zh-latest": (RAIN_ALLEY, RAIN_ALLEY_SHOTS),
}


def example_visual_plan(lang):
    subjects, beats = EXAMPLE_VISUALS[lang]
    return {**subjects, "shots": [
        {"cue": cue, "offset": 0, "transition": transition, "beat": beat, "prompt": prompt, "effects": effects}
        for cue, transition, beat, prompt, *effects in beats
    ]}
