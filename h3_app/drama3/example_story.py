"""Comprehensive showcase story for Soundstage (Drama 3).

Demonstrates multi-scene spatial staging (`room`, `stage`), ambience and music loops,
panned and moving one-shot sound effects, musical stingers, character/device/place vocal
effects, `<near>` and `<far>` voice distance, simultaneous overlap, listener reactions,
barge-in pauses (`<#-0.3#>`), cut-off dashes (`—`), all seven scoped delivery tags,
audible events, and Arpabet pronunciation notes.
"""

from __future__ import annotations

import json

EXAMPLE_STORY_TITLE = "The Beacon of Blackthorn Reef"

EXAMPLE_STORY_SPEAKERS: tuple[dict, ...] = (
    {
        "speaker_id": 0,
        "name": "narrator",
        "description": (
            "A male voice: mature, late forties, warm baritone with resonant chest tone, "
            "crisp studio articulation, and a measured, atmospheric storytelling pace. "
            "Clean dry studio recording, no background noise."
        ),
        "preview_text": (
            "The lighthouse at Blackthorn Reef had stood empty for thirty years, its great "
            "lantern dark above the Atlantic swells. When the harbour master sent us out "
            "before the autumn equinox, nobody expected the machinery inside the tower to "
            "still be warm."
        ),
    },
    {
        "speaker_id": 1,
        "name": "Elena",
        "description": (
            "A female voice: early thirties, clear mezzo-soprano, focused and observant "
            "marine investigator with steady breath control, natural conversational pacing, "
            "and crisp consonants. Clean dry studio recording."
        ),
        "preview_text": (
            "I checked the tide charts twice before we left the mainland pier this morning. "
            "The cable trench runs straight from the breakwater into the base of the reef, "
            "and somebody has been drawing power from it all week."
        ),
    },
    {
        "speaker_id": 2,
        "name": "Marcus",
        "description": (
            "A male voice: mid-sixties, weathered gravelly bass-baritone, former lighthouse "
            "keeper with a rough coastal cadence, unhurried delivery, and grounded authority. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "You can read all the blueprints you want back in the harbour office, miss, "
            "but granite and salt water keep their own ledger. When the gale comes around "
            "the point, this tower hums like a tuning fork."
        ),
    },
    {
        "speaker_id": 3,
        "name": "Vance",
        "description": (
            "A male voice: early forties, brisk tenor, coastal dispatch officer with "
            "clipped professional diction, alert energy, and an urgent operational cadence. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "Harbour control to survey launch, we are tracking a steep barometer drop "
            "twenty miles west of the shoals. Keep your channel open on sixteen and report "
            "as soon as you reach the landing."
        ),
    },
    {
        "speaker_id": 4,
        "name": "Aegis",
        "description": (
            "An androgynous voice: calm, precision-engineered mid-register voice with even "
            "syllable weight, cool synthetic clarity, and unshakeable composure. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "Primary telemetry remains online across all acoustic arrays in the lower "
            "sector. Harmonic calibration will resume once the outer pressure seals are "
            "locked and verified."
        ),
    },
    {
        "speaker_id": 5,
        "name": "Siren",
        "description": (
            "A female voice: timeless, airy alto with velvet resonance, slow hypnotic "
            "phrasing, and delicate breathy overtones. Clean dry studio recording."
        ),
        "preview_text": (
            "Beneath the white foam and the iron chains, the water remembers every ship "
            "that ever crossed the bar. Listen closely to the tide, and you will hear the "
            "bells still ringing in the deep."
        ),
    },
)

EXAMPLE_STORY_SCRIPT = """\
narrator: The Beacon of Blackthorn Reef.
<#0.6#>
<parallel room="small_room" stage="Elena=-0.4, Marcus=0.4">
  <sound loop="true" volume="0.35" duration="25" name="coastal_storm">Heavy coastal rain and wind lashing against thick lighthouse windows, distant ocean surf</sound>
  <sound loop="true" volume="0.2" duration="36" name="noir_cello">Slow brooding minor-key solo cello and muted piano pulse, dark maritime mystery, no vocals</sound>
  <sequential>
    <parallel>
      <sound name="iron_door" duration="3" pan="-0.6">Heavy rusted iron door creaking open and clanging shut against stone</sound>
      narrator: Rain hammered the lantern glass [tense, low] as Elena forced the watchroom door shut.
    </parallel>
    <parallel>
      <sound name="approaching_boots" duration="4" pan="0.8,-0.2">Heavy wet rubber boots walking across wooden floorboards</sound>
      narrator: Marcus crossed the timber floor [slowly, heavy] and set his lantern beside the brass console.
    </parallel>
    Elena: Look at the voltmeter, [quietly amazed] <emphasis>Marcus</emphasis>.
    Elena: Siobhan{SH IH0 V AO1 N} swore this station was decommissioned, [focused, analytical] yet the main bus is <stress>fully</stress> charged.
    <parallel>
      Marcus: We shut the diesel generators down [gravelly, uneasy] thirty years ago, <slow>girl</slow>. Whatever is feeding these coils isn't coming from the mainland grid.
      <sequential>
        <#1.4#>
        Elena: Wait. [gasps]
      </sequential>
    </parallel>
    Elena: <near><intimate_whisper> Listen to the casing, [hushed, intense] it's <whisper>breathing</whisper> under the floor.
    <parallel>
      <sound name="radio_static" duration="3" pan="0.4">Harsh marine VHF radio squelch burst and crackling static dropout</sound>
      narrator: On the shelf, the marine transceiver [sudden, sharp] crackled to life.
    </parallel>
    Vance: <walkie_talkie> Mainland dispatch to Blackthorn team, [urgent, brisk] come in! The storm front is <fast>closing the channel</fast> ahead of schedule.
    Elena: Copy, Vance, [calling out clearly] we're inside the lower watchroom, but we found an active <emphasis>signal array</emphasis>.
    Vance: <bad_signal> Get out of the tower, [panicked, breaking up] the sonar buoys just registered a <forceful>massive spike</forceful> right under the reef! [coughs]
  </sequential>
</parallel>
<#1.0#>
<parallel room="outdoors" stage="Elena=-0.5, Marcus=0.5">
  <sound loop="true" volume="0.4" duration="25" name="coastal_storm">Heavy coastal rain and wind lashing against thick lighthouse windows, distant ocean surf</sound>
  <sequential>
    narrator: They burst onto the outer stone gallery, [sweeping, dramatic] where gale-force spray tore across the <slow>catwalk</slow>.
    Marcus: <far> Hold the iron rail, [shouting over wind] <forceful>Elena</forceful>, the surge is breaking over the shoal!
    <#-0.3#>
    Elena: Look at the water, [breathless, stunned] the waves are <emphasis>standing still</emphasis> in a circle!
    <parallel>
      <sound name="deep_bell" duration="5" distance="far">Enormous bronze fog bell tolling once in the distance over stormy water</sound>
      narrator: Far out beyond the white foam, [solemn, eerie] a submerged bell tolled through the <soft>darkness</soft>.
    </parallel>
    <parallel>
      Elena: Above us!
      Marcus: The lantern!
    </parallel>
  </sequential>
</parallel>
<#1.0#>
<parallel room="large_hall" stage="Elena=-0.35, Marcus=0.35, Aegis=0.0">
  <sound loop="true" volume="0.35" duration="25" name="chamber_hum">Deep low-frequency mechanical hum and resonant stone room tone with slow water drips</sound>
  <sound loop="true" volume="0.2" duration="40" name="tension_ostinato">Tense pulsing synth bass and bowed contrabass ostinato, slow suspenseful build, no vocals</sound>
  <sequential>
    <parallel>
      <sequential>
        <sound name="iron_door" duration="3" pan="-0.6">Heavy rusted iron door creaking open and clanging shut against stone</sound>
        <sound name="glass_shatter" duration="3" pan="-0.5" distance="near">Lantern glass shattering sharply onto a stone floor</sound>
      </sequential>
      narrator: Inside the vaulted lantern chamber, [tense, echoing] a cracked prism fell shattering across the stone floor.
    </parallel>
    Aegis: <robotic_voice> Acoustic lock confirmed, [icy calm, steady] <emphasis>keeper</emphasis> recognized.
    Marcus: That voice, [shaken, disbelieving] we sealed the <stress>Aegis core</stress> behind lead bulkheads —
    <parallel>
      Marcus: when the old ship went down! [shaky breath]
      <sequential>
        <#0.4#>
        Elena: Wait, [turning on him, accusing] you <emphasis>knew</emphasis> it was down here?
      </sequential>
    </parallel>
    Marcus: We had no choice, [heavy with regret] the beacon wasn't built to guide ships, <slow>Elena</slow>. [sighs]
    <parallel>
      Aegis: <robotic_voice> Deep trench receiver opening now, [measured, relentless] relaying <stress>hydrophone</stress> channel zero.
      <sequential>
        <#1.1#>
        <sound name="radio_static" duration="3" pan="0.4">Harsh marine VHF radio squelch burst and crackling static dropout</sound>
      </sequential>
    </parallel>
  </sequential>
</parallel>
<#1.2#>
<parallel room="underwater">
  <sound loop="true" volume="0.35" duration="25" name="abyssal_current">Deep underwater current with muffled low-frequency pressure groan and rising air bubbles</sound>
  <sound loop="true" volume="0.2" duration="40" name="ethereal_choir_pad">Shimmering glass harmonica and warm ambient synth pad, solemn and transcendent, no vocals</sound>
  <sequential>
    narrator: Through the acoustic lens, [hushed, spellbound] the chamber filled with the sound of the <slow>flooded vault</slow> far below the reef.
    Siren: <ghost_spirit><far> Three decades of silence, [distant, hypnotic] and still the <soft>keepers</soft> return to the light.
    Elena: <near> Who is speaking, [whispering, awed] is that coming from <whisper>beneath the rock</whisper>?
    Siren: <ghost_spirit><near> Do not fear the tide, [warm, otherworldly] we only held the <emphasis>memory</emphasis> Siobhan{SH IH0 V AO1 N} left in the deep.
  </sequential>
</parallel>
<#0.8#>
Aegis: <robotic_voice><reverb_large_hall> Ignition sequence complete, [solemn, resonant] the <emphasis>beacon</emphasis> is online.
<parallel>
  <sound name="revelation_sting" volume="0.4" duration="6">Dramatic brass and timpani swell with high shimmering strings, single cinematic hit, no vocals</sound>
  narrator: A blinding beam of white light [warm, triumphant] swept out across the storm, <slow>parting the dark sea</slow>.
</parallel>
Marcus: Look at that horizon, [quiet wonder, relieved] <soft>still</soft> burning after all these years. [chuckles]"""


def example_story_reply_json(lang: str = "en") -> str:
    """The example story packaged in the writer's JSON reply format."""
    if str(lang or "en").lower().startswith("zh"):
        return json.dumps(
            {
                "title": EXAMPLE_STORY_ZH_TITLE,
                "source_mode": "brief",
                "script": EXAMPLE_STORY_ZH_SCRIPT,
                "speakers": list(EXAMPLE_STORY_ZH_SPEAKERS),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "title": EXAMPLE_STORY_TITLE,
            "source_mode": "brief",
            "script": EXAMPLE_STORY_SCRIPT,
            "speakers": list(EXAMPLE_STORY_SPEAKERS),
        },
        ensure_ascii=False,
    )


EXAMPLE_STORY_ZH_TITLE = "昆仑墟底的回响"

EXAMPLE_STORY_ZH_SPEAKERS: tuple[dict, ...] = (
    {
        "speaker_id": 0,
        "name": "narrator",
        "description": (
            "A male voice: mature Mandarin narrator in his early forties, deep warm baritone, "
            "calm and magnetic resonance, clear standard articulation, and a steady cinematic "
            "storytelling pace. Clean dry studio recording, no background noise."
        ),
        "preview_text": (
            "昆仑山脉的暴风雪已经持续了整整三天，营地外到处是结冰的岩壁。当科考队推开那扇埋在冰层下的"
            "青铜大门时，谁也没想到地底深处还留着千年前的灯火。"
        ),
    },
    {
        "speaker_id": 1,
        "name": "沈星河",
        "description": (
            "A female voice: late twenties Mandarin speaker, clear and composed mezzo-soprano, "
            "sharp analytical tone, crisp diction, and a steady, focused conversational pace. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "我已经把三组地震波传感器的读数全部核对了一遍。冰层下方的空腔结构非常规整，那些回声绝不是"
            "天然形成的裂谷，我们要找的观测台就在正下方。"
        ),
    },
    {
        "speaker_id": 2,
        "name": "尉迟烈",
        "description": (
            "A male voice: late forties Mandarin speaker, rugged gravelly bass-baritone, "
            "seasoned expedition captain with grounded authority, firm breath support, and "
            "a steady unhurried cadence. Clean dry studio recording."
        ),
        "preview_text": (
            "在高原雪线上走了二十年，什么古怪的地形我都见过。大家都把安全绳扣紧，检查好氧气瓶的压力表，"
            "没有我的命令谁也不准擅自离开主通道。"
        ),
    },
    {
        "speaker_id": 3,
        "name": "周岚",
        "description": (
            "A female voice: mid-thirties Mandarin speaker, brisk alto, base camp radio "
            "dispatcher with alert energy, clean crisp pronunciation, and an efficient "
            "operational rhythm. Clean dry studio recording."
        ),
        "preview_text": (
            "这里是昆仑一号地面中继站，目前高空电离层非常不稳定。请各小组保持短波频道畅通，每隔十五分钟"
            "向大本营汇报一次深度和坐标。"
        ),
    },
    {
        "speaker_id": 4,
        "name": "天枢",
        "description": (
            "An androgynous voice: Mandarin speaker, cool metallic mid-register voice with "
            "even syllable weight, calm precision, and unshakeable synthetic clarity. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "天枢星轨阵列已完成自检，地脉共振频率维持在正常区间。待外层封门完全闭合后，主星图将自动"
            "恢复推演。"
        ),
    },
    {
        "speaker_id": 5,
        "name": "守墟古灵",
        "description": (
            "A female voice: timeless Mandarin alto with deep ethereal resonance, slow solemn "
            "phrasing, and gentle breathy overtones. Clean dry studio recording."
        ),
        "preview_text": (
            "星辰升起又落下，冰川掩埋了昔日的城池与歌谣。唯有刻在青铜上的誓言，还在漫长的黑夜里等待"
            "归来的人。"
        ),
    },
)

EXAMPLE_STORY_ZH_SCRIPT = """\
narrator: 昆仑墟底的回响。
<#0.6#>
<parallel room="small_room" stage="沈星河=-0.4, 尉迟烈=0.4">
  <sound loop="true" volume="0.35" duration="25" name="glacier_blizzard">High altitude blizzard wind howling against metal shelter walls, creaking ice</sound>
  <sound loop="true" volume="0.2" duration="36" name="guqin_mystery">Sparse deep guqin plucks and low bowed cello drone, ancient mystery, no vocals</sound>
  <sequential>
    <parallel>
      <sound name="bronze_seal" duration="4" pan="-0.6">Heavy ancient bronze mechanism unlocking with deep metallic clunk and stone scrape</sound>
      narrator: 暴风雪拍打着岩壁，[tense, low] 沈星河用力合上了前哨站的青铜闸门。
    </parallel>
    <parallel>
      <sound name="snow_boots" duration="4" pan="0.8,-0.2">Heavy mountaineering boots crunching on frost and stepping onto stone floor</sound>
      narrator: 尉迟{yu4 chi2}烈踏着满地寒霜走来，[slowly, heavy] 把探照灯放在石台边缘。
    </parallel>
    沈星河: 你看石盘上的刻度，[quietly amazed] <emphasis>尉迟{yu4 chi2}队长</emphasis>。
    沈星河: 碑文记载二十八宿{xiu4}早已停转，[focused, analytical] 可地脉指针却<stress>完全</stress>对准了北极星。
    <parallel>
      尉迟烈: 这座观测台封存了整整一千年，[gravelly, uneasy] <slow>星河</slow>。驱动齿轮的绝不是地面上的风力。
      <sequential>
        <#1.3#>
        沈星河: 等等。[gasps]
      </sequential>
    </parallel>
    沈星河: <near><intimate_whisper> 你听底座下面，[hushed, intense] 好像有什么东西在<whisper>轻轻呼吸</whisper>。
    <parallel>
      <sound name="shortwave_burst" duration="3" pan="0.4">Shortwave military radio squelch and high-altitude electromagnetic static crackle</sound>
      narrator: 桌角的短波电台，[sudden, sharp] 突然炸开一阵刺耳的电流声。
    </parallel>
    周岚: <walkie_talkie> 大本营呼叫前哨组，[urgent, brisk] 收到请回答！冰川下方的磁场正在<fast>急剧翻转</fast>！
    沈星河: 周岚，我们听得见，[calling out clearly] 先锋队刚进入外层石室，正准备测算<emphasis>共振源</emphasis>。
    周岚: <bad_signal> 立刻撤出冰缝，[panicked, breaking up] 卫星监测到地底出现了<forceful>超高频脉冲</forceful>！[coughs]
  </sequential>
</parallel>
<#1.0#>
<parallel room="outdoors" stage="沈星河=-0.5, 尉迟烈=0.5">
  <sound loop="true" volume="0.4" duration="25" name="glacier_blizzard">High altitude blizzard wind howling against metal shelter walls, creaking ice</sound>
  <sequential>
    narrator: 两人冲出悬崖栈道，[sweeping, dramatic] 狂风卷着冰碴在深渊上方<slow>呼啸盘旋</slow>。
    尉迟烈: <far> 抓紧铁索，[shouting over wind] <forceful>星河</forceful>，脚下的冰梁快撑不住了！
    <#-0.3#>
    沈星河: 快看深渊底下，[breathless, stunned] 整片暗河都在<emphasis>发光</emphasis>！
    <parallel>
      <sound name="ancient_chime" duration="5" distance="far">Deep ancient bronze bell tolling once in a distant mountain chasm</sound>
      narrator: 万丈冰渊深处，[solemn, eerie] 传来一声悠远苍凉的<soft>青铜钟鸣</soft>。
    </parallel>
    <parallel>
      沈星河: 在上面！
      尉迟烈: 浑天仪！
    </parallel>
  </sequential>
</parallel>
<#1.0#>
<parallel room="large_hall" stage="沈星河=-0.35, 尉迟烈=0.35, 天枢=0.0">
  <sound loop="true" volume="0.35" duration="25" name="cavern_resonance">Subterranean stone hall resonance with low bronze gear hum and slow water drips</sound>
  <sound loop="true" volume="0.2" duration="40" name="taiko_suspense">Slow tense low string ostinato with deep muffled frame drum pulse, cinematic suspense, no vocals</sound>
  <sequential>
    <parallel>
      <sequential>
        <sound name="bronze_seal" duration="4" pan="-0.6">Heavy ancient bronze mechanism unlocking with deep metallic clunk and stone scrape</sound>
        <sound name="jade_shatter" duration="3" pan="-0.5" distance="near">Ancient jade tablet cracking and shattering sharply onto stone tiles</sound>
      </sequential>
      narrator: 空旷宏大的青铜大殿内，[tense, echoing] 一块古玉圭应声跌碎在石砖上。
    </parallel>
    天枢: <robotic_voice> 星轨锁已解除，[icy calm, steady] 确认<emphasis>司天监</emphasis>印信。
    尉迟烈: 这声音不对，[shaken, disbelieving] 当年先祖明明已经把<stress>天枢核心</stress>彻底封死——
    <parallel>
      尉迟烈: 绝不可能自己醒过来！[shaky breath]
      <sequential>
        <#0.4#>
        沈星河: 慢着，[turning on him, accusing] 你早就<emphasis>知道</emphasis>天枢的存在？
      </sequential>
    </parallel>
    尉迟烈: 我也是奉命行事，[heavy with regret] 尉迟{yu4 chi2}一族世代守山，就是为了把星图还{huan2}回<slow>地心</slow>。[sighs]
    <parallel>
      天枢: <robotic_voice> 寒渊听音阵列正在开启，[measured, relentless] 即将接通<stress>零号地脉</stress>回波。
      <sequential>
        <#1.1#>
        <sound name="shortwave_burst" duration="3" pan="0.4">Shortwave military radio squelch and high-altitude electromagnetic static crackle</sound>
      </sequential>
    </parallel>
  </sequential>
</parallel>
<#1.2#>
<parallel room="underwater">
  <sound loop="true" volume="0.35" duration="25" name="subglacial_river">Deep subglacial underwater current with low pressure rumble and rising bubbles</sound>
  <sound loop="true" volume="0.2" duration="40" name="celestial_pad">Warm ethereal xiao flute overtones and shimmering ambient synth pad, solemn and vast, no vocals</sound>
  <sequential>
    narrator: 随着水镜缓缓亮起，[hushed, spellbound] 大殿里回荡起冰河深处传来的<slow>古老脉动</slow>。
    守墟古灵: <ghost_spirit><far> 千年风雪已过，[distant, hypnotic] 守望星河的人终于<soft>归来</soft>。
    沈星河: <near> 是谁在说话，[whispering, awed] 声音是从<whisper>寒渊水底</whisper>传上来的吗？
    守墟古灵: <ghost_spirit><near> 不必惊慌，[warm, otherworldly] 我们只是替尉迟{yu4 chi2}先人守住了这缕<emphasis>星火</emphasis>。
  </sequential>
</parallel>
<#0.8#>
天枢: <robotic_voice><reverb_large_hall> 浑天星图校准完毕，[solemn, resonant] 昆仑<emphasis>主烽燧</emphasis>已点亮。
<parallel>
  <sound name="starlight_sting" volume="0.4" duration="6">Grand cinematic orchestral brass and chime swell, single triumphant hit, no vocals</sound>
  narrator: 一道璀璨的苍青色光柱，[warm, triumphant] 冲破漫天风雪，<slow>直贯万里云霄</slow>。
</parallel>
尉迟烈: 整整一千年了，[quiet wonder, relieved] 这座山终于又<soft>活过来</soft>了。[chuckles]"""
