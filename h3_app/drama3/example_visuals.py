"""Illustration beats for the three built-in Soundstage scripts, on their dialogue cues."""

BLACKTHORN = {
    "characters": {
        "elena": "Elena, observant woman in her thirties, dark hair in a low ponytail, navy storm parka and brass compass",
        "marcus": "Marcus, broad, weathered lighthouse keeper in his sixties, grey beard and olive oilskin coat",
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
     "Wide establishing view of @reef(unlit crown), black rocks breaking the slate-blue sea. Hold the tower silhouette clear against layered storm clouds.", "rain", "push_in"),
    (1, "cut", "Elena wrestles the watchroom door shut.",
     "Eye-level medium shot of @elena(both hands on the heavy door) inside @watchroom, a narrow strip of cold exterior light at the threshold."),
    (2, "cut", "Marcus brings his lantern to the console.",
     "Low medium view of @marcus(setting @lantern beside @console) in @watchroom, an amber pool on the wet boards.", "flicker"),
    (3, "cut", "The voltmeter is still live.",
     "Insert of @console(needle pinned high) in @watchroom, the curved glass catching a reflection from @lantern.", "push_in"),
    (6, "cut", "Elena feels a pulse under the floor.",
     "Floor-level close shot of @elena(kneeling, palm against @console), a ring of droplets trembling on the casing; @marcus stands behind her in shadow.", "subtle_sway"),
    (8, "cut", "The transceiver breaks the silence.",
     "Tight view of @radio(indicator illuminated) on the shelf of @watchroom, coiled cable trailing toward the edge.", "vignette"),
    (10, "cut", "Vance's warning stops Elena mid-report.",
     "Two-shot in @watchroom: @elena(holding the radio handset close) looks sharply at @marcus, who has already turned toward the stair.", "cold_tint", "subtle_sway"),
    (12, "cut", "They step onto the exposed gallery.",
     "Wide low angle of @elena and @marcus(braced against the gale) emerging onto @gallery, @lantern a small amber point against the sea.", "rain", "pan_right", "cold_tint"),
    (14, "cut", "The waves stand in a ring.",
     "High view from @gallery toward the black water, a continuous rim of white foam around the eerily smooth center.", "pull_out"),
    (16, "cut", "Elena and Marcus look up at the lantern crown.",
     "From rail height, @elena and @marcus look overhead toward @reef(crown framed by the upper gallery), wet iron bars in the foreground.", "rain", "lightning", "pan_up"),
    (18, "cut", "A prism shatters on the lantern room floor.",
     "Low close shot in @chamber of a broken prism at the foot of @core, its brass rings rising into darkness.", "impact_shake"),
    (19, "cut", "The voice of Aegis recognizes the keeper.",
     "Reverse medium shot in @chamber: @elena turns toward @marcus as he stares into @core. Keep both faces readable in cold window light."),
    (23, "dissolve", "Aegis opens the receiver to the flooded vault.",
     "Distant view through @vault with silvery bubbles below the arches and a faint cyan glow escaping the bulkhead seam.", "fog", "push_in", "cold_tint"),
    (26, "dissolve", "Elena listens for the voice below the rock.",
     "Close profile of @elena beside @core in @chamber, teal reflections on her cheek, iron window tracery softened behind her.", "dream_bloom"),
    (28, "cut", "The beacon ignites.",
     "Low wide view inside @chamber, @core opening to white-gold light as @elena and @marcus shield their eyes below its rings.", "flash", "pull_out"),
    (29, "cut", "The beam crosses the storm-dark sea.",
     "Wide exterior of @reef(beacon lit), a clean beam out over the swells, white foam giving way to the quiet horizon.", "pull_out"),
)

KUNLUN = {
    "characters": {
        "xinghe": "沈星河, young Chinese expedition researcher, short black hair, rust-red insulated jacket and grey gloves",
        "yuchi": "尉迟烈, older Chinese field commander, weathered face, dark indigo expedition coat and fur-lined hood",
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
     "Wide silhouette of @ridge with one small outpost light in the blue twilight. Reserve the foreground for sculpted rock and snow ridges.", "snow", "push_in"),
    (1, "cut", "沈星河合上青铜闸门。",
     "Medium interior angle in @outpost, @xinghe braces both shoulders against the bronze gate while snow collects at the narrowing threshold."),
    (2, "cut", "探照灯照亮冰霜和石盘。",
     "Low shot of @yuchi(setting @lamp on the bench beside @dial) in @outpost; warm light reaches across the frost.", "warm_glow", "dust"),
    (3, "cut", "指针指向北极星。",
     "Overhead insert of @dial(pointer aligned with the outer star carving), @xinghe's gloved fingertips hovering above the ring.", "push_in"),
    (7, "cut", "底座传来微弱的呼吸。",
     "Ground-level close view of @xinghe listening beside @dial, grains of frost shifting in the seam beneath the base.", "vignette", "subtle_sway"),
    (8, "cut", "短波电台突然响起。",
     "Close view of @radio(status light on) on the stone workbench of @outpost, edge-lit by @lamp."),
    (11, "cut", "撤退的警告让两人停住。",
     "Two-shot of @xinghe gripping the handset and @yuchi turning toward the gate inside @outpost. Keep the cold gate beyond their shoulders."),
    (12, "cut", "两人踏上悬崖栈道。",
     "Wide shot of @xinghe and @yuchi following the chain across @walkway. Leave most of the lower frame to the drop.", "snow", "pan_left", "cold_tint"),
    (14, "cut", "深渊里的暗河亮起来。",
     "Steep overhead view from @walkway toward @ice_river, a narrow ribbon of blue light far below fractured ice shelves.", "pan_down", "cold_tint"),
    (16, "cut", "栈道上方现出浑天仪。",
     "Low upward angle on @xinghe and @yuchi looking toward the outline of @armillary inside a high opening cut into the cliff.", "fog", "pan_up"),
    (18, "cut", "玉圭在青铜大殿内摔碎。",
     "Low close shot of pale jade fragments across the dark tiles of @hall, @armillary reflected faintly in the floor.", "impact_shake"),
    (19, "cut", "天枢回应，尉迟烈说出实情。",
     "Medium two-shot beside @armillary in @hall. @xinghe watches @yuchi, whose face catches light reflected from the bronze rings."),
    (23, "dissolve", "水镜接通寒渊的回波。",
     "Close view of @mirror on the floor of @hall, cyan ripples spreading out from a single point, the bronze rim sharp behind them.", "push_in", "dream_bloom"),
    (25, "dissolve", "冰河深处传来守墟古灵的声音。",
     "View under the translucent roof of @ice_river with a small star-like light beyond the layered ice, no figure visible.", "fog", "dream_bloom"),
    (28, "cut", "天枢点亮主烽燧。",
     "Symmetrical wide interior of @hall, @armillary aligned around a bright cyan center, @xinghe and @yuchi in silhouette below.", "flash", "push_in"),
    (29, "cut", "苍青光柱穿出昆仑风雪。",
     "Distant wide shot of @ridge with a narrow shaft of cyan light rising from the summit through windblown snow.", "snow", "pull_out", "cold_tint"),
)

RAIN_ALLEY = {
    "characters": {
        "achuan": "阿川, lanky young Chinese man with tousled black hair and an oversized faded green rain jacket",
        "zhixia": "许知夏, young Chinese clockmaker's apprentice, black bob, navy work apron over burgundy knitwear",
        "chen": "陈伯, elderly Chinese watchmaker with silver hair and wire-rimmed glasses, brown cardigan and rolled cuffs",
        "lin": "林姨, older Chinese woman, neatly pinned grey hair, plum raincoat and woven shopping basket",
    },
    "locations": {
        "street": "narrow old neighborhood lane, tiled eaves, wet paving and soft yellow shop lights at dusk",
        "shop": "small clock repair shop, green awning, oak counter, clocks on the rear wall and a low street-facing sill",
    },
    "props": {
        "radio": "old cream portable radio with a black dial, brass speaker grille and loose battery cover",
        "note": "small timetable note folded several times, ink marks too small to read",
        "mic": "round silver tabletop broadcast microphone with a worn cloth lead",
        "clock": "slim walnut wall clock with a brass pendulum behind beveled glass",
        "tickets": "two pale paper train tickets, creased at the corners with no legible lettering",
    },
}
RAIN_ALLEY_SHOTS = (
    (0, "cut", "小雨落在老街的屋檐下。",
     "Wide blue-hour view of @street with one warm window at @shop. Wet paving reflects the window along the empty lane.", "rain", "push_in"),
    (1, "cut", "阿川抱着收音机钻进雨棚。",
     "At awning height, @achuan hurries into the dry patch outside @shop carrying @radio. Keep the rain behind him in the street.", "rain", "pan_right"),
    (3, "cut", "知夏指给阿川看空着的电池槽。",
     "Over-counter two-shot of @zhixia pointing at @radio(battery cover loose) while @achuan gives her an embarrassed grin."),
    (7, "cut", "陈伯攥着折旧的时刻表。",
     "Medium view through @shop: @zhixia passes over a battery in the foreground while @chen grips @note at the oak counter.", "push_in"),
    (11, "cut", "打烊后，陈伯把纸摊平。",
     "Close shot of @chen's hands unfolding @note on the oak counter of @shop, with a blurred closed sign beyond.", "warm_glow"),
    (13, "cut", "那只钟早就修好了。",
     "Insert of @clock on the wall of @shop, polished pendulum still, other clocks falling out of focus behind it.", "dust"),
    (16, "cut", "陈伯说起那些拖延的日子。",
     "Eye-level profile of @chen beside @clock, eyes lowered as @zhixia listens across the counter. Use the work lamp as the only warm accent.", "push_in", "dream_bloom"),
    (20, "cut", "知夏提醒师父，林姨一直懂他。",
     "Reverse shot of @zhixia facing @chen with both hands on the counter. Keep the damp blue window softly visible."),
    (23, "cut", "年轻人把话筒递到陈伯面前。",
     "Medium three-shot in @shop: @zhixia offers @mic to @chen while @achuan places @radio on the sill."),
    (27, "dissolve", "七点整，收音机亮了。",
     "Close on @radio glowing inside the window of @shop, rain streaking the glass behind it and @mic out of focus in the foreground.", "warm_glow", "push_in", "rain"),
    (28, "cut", "陈伯开口，请林姨留一个座位。",
     "Intimate profile of @chen speaking into @mic with his eyes fixed on the open doorway, the clock faces quiet behind him.", "push_in"),
    (32, "cut", "门外传来篮子碰门槛的轻响。",
     "Low shot across the threshold of @shop: @lin's basket touches the stone as she pauses just outside, wet lane behind her.", "subtle_sway"),
    (33, "cut", "林姨走进灯下，陈伯转过身。",
     "Medium two-shot of @lin in the doorway and @chen turning toward her, @mic resting between them on the counter.", "warm_glow"),
    (35, "cut", "林姨拿出早已买好的两张车票。",
     "Close shot of @lin placing @tickets into @chen's open palm, focus on the hands and the worn corners of the tickets.", "film_grain"),
    (40, "cut", "陈伯轻轻推动钟摆。",
     "Close-up at counter height of @chen's fingers releasing the brass pendulum of @clock, with @lin reflected faintly in the glass.", "push_in"),
    (42, "dissolve", "两人准备一起出门。",
     "Wide view from @street into @shop, @chen and @lin standing together at the threshold, @zhixia and @achuan in the warm room behind them.", "rain", "pull_out", "warm_glow"),
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
