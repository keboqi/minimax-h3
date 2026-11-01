"""Chinese showcase for the shared Single-pass and Guided production controls.

Three scenes move from a lively market through a quiet confession to a reunion.
English performance cues and acoustic prompts are intentional model inputs;
the title, dialogue and neutral voice-reference samples are Chinese.
"""

EXAMPLE_STORY_ZH_LATEST_TITLE = "雨巷里的末班广播"

EXAMPLE_STORY_ZH_LATEST_SPEAKERS: tuple[dict, ...] = (
    {
        "speaker_id": 0,
        "name": "narrator",
        "description": (
            "A female voice: Mandarin narrator in her forties, warm mid-register alto, "
            "smooth unforced resonance, precise standard Mandarin articulation and an "
            "unhurried observant storytelling rhythm. Clean dry studio speech, no music or noise."
        ),
        "preview_text": (
            "老街的雨通常在傍晚停下，石板路却要到第二天才干。沿街的店铺陆续亮灯，卖点心的人收好竹篮，"
            "修理铺门前还留着一张供街坊歇脚的长凳。"
        ),
    },
    {
        "speaker_id": 1,
        "name": "许知夏",
        "description": (
            "A female voice: Mandarin speaker in her late twenties, clear medium-high "
            "mezzo-soprano, lightly textured natural timbre, neat consonants and a measured "
            "conversational rhythm, attentive and practical. Clean dry studio speech, no music or noise."
        ),
        "preview_text": (
            "这台收音机的旋钮已经换好了，原来的木头外壳还很结实。我把电池放在柜台左边，等天晴以后，"
            "再把后盖拆下来清理一遍，平时照常使用就可以。"
        ),
    },
    {
        "speaker_id": 2,
        "name": "阿川",
        "description": (
            "A male voice: Mandarin speaker in his early twenties, bright mid-register tenor, "
            "slightly airy timbre, nimble clear articulation and an easy conversational pace, "
            "outgoing without exaggerated energy. Clean dry studio speech, no music or noise."
        ),
        "preview_text": (
            "我每天送完最后一趟货，都会从河边这条路回来。桥头的路灯已经修好，晚上推车也看得清楚，"
            "你要是带的东西多，我可以顺路帮你送到店门口。"
        ),
    },
    {
        "speaker_id": 3,
        "name": "陈伯",
        "description": (
            "A male voice: Mandarin speaker in his seventies, low rounded bass-baritone, "
            "soft dry grain, careful consonants and a steady deliberate pace with grounded "
            "gentleness. Clean dry studio speech, no device filtering, music or noise."
        ),
        "preview_text": (
            "以前修东西，总要先把每一颗螺丝收进小盒子里。等到装回去的时候，按顺序拿出来就不会弄错，"
            "这是个慢办法，不过用了几十年，我已经习惯了。"
        ),
    },
    {
        "speaker_id": 4,
        "name": "林姨",
        "description": (
            "A female voice: Mandarin speaker in her late sixties, full middle-low alto, "
            "slightly husky warm resonance, firm clear articulation and an even practical "
            "rhythm, direct and caring. Clean dry studio speech, no music or noise."
        ),
        "preview_text": (
            "今天的面汤放在灶上温着，回来以后热一下就能吃。门口那把旧伞我已经补过了，伞骨还算牢靠，"
            "下雨出门记得带着，用完放在檐下晾干。"
        ),
    },
)

EXAMPLE_STORY_ZH_LATEST_SCRIPT = """\
narrator: 雨巷里的末班广播。
<parallel stage="许知夏=-0.35, 阿川=0.4, 陈伯=-0.1" holds="连电池都没装~0.6">
  <sound name="market_rain" loop="true" volume="0.32" duration_class="texture" role="social" perspective="in_world">Light rain on a canvas market awning, scattered indistinct vendor murmurs without intelligible words, occasional distant bicycle wheels, relaxed evening street</sound>
  <sound name="radio_switch" duration_class="accent" role="physical" perspective="in_world" anchor="开给你听" edge="end" pan="0.4">One close dry click of a stiff plastic switch on an old portable radio</sound>
  <sound name="empty_battery_pluck" volume="0.3" duration_class="accent" role="comic" perspective="editorial" anchor="连电池都没装" edge="end">Single quiet muted pizzicato string pluck, gently deflated playful note with quick natural decay, no vocals</sound>
  <sequential>
    narrator: 六点五十分，老街还落着小雨，[warm, observant] 阿川抱着一台旧收音机，钻进陈伯钟表铺的<soft>雨棚底下</soft>。
    阿川: 知夏，你瞧，[trying to impress her] 这台一开，全街的消息，我都能<emphasis>第一个听见</emphasis>。
    许知夏: 你先别吹，[dry, gently teasing] 师父修了半天的机器，你连<stress>电池盖</stress>都没装回去。
    阿川: 那是留着检查的，[confident, covering embarrassment] 不信，我<emphasis>开给你听</emphasis>。
    许知夏: 别拧了，[fond, puncturing his boast] 你<stress>连电池都没装</stress>。
    阿川: 我说怎么这么安静，[sheepish, conceding to her] 原来好消息还得<soft>自己往里放</soft>。[small laugh]
    narrator: 许知夏把电池递给他，[quiet, noticing a change] 转头看见师父站在柜台后，攥着一张<slow>折了又折的纸</slow>。
    陈伯: 小许，[careful, asking a favor] 七点那趟街坊广播，能不能让我<emphasis>说两句</emphasis>？
    许知夏: 能啊，[gentle, checking with him] 不过明天就拆广播线了，您要通知谁，今晚可得<stress>说清楚</stress>。
    陈伯: 不通知谁，[trying to keep composure] 我就是有句话，<soft>欠得太久了</soft>。
  </sequential>
</parallel>
<#0.8#>
<parallel stage="许知夏=-0.3, 陈伯=0.2, 阿川=0.45" holds="早就修好了~0.6; 我没敢答应~1.2">
  <sound name="shop_rain" loop="true" volume="0.25" duration_class="texture" role="physical" perspective="in_world">Soft rain dripping from a shop awning just outside an open doorway, quiet distant street wash, no voices or machinery</sound>
  <sound name="letter_unfold" duration="2" role="physical" perspective="in_world" anchor="把纸慢慢摊平" edge="start" pan="0.2">A creased sheet of thin paper slowly unfolded and smoothed on a wooden counter, close dry rustle</sound>
  <sound name="homeward_piano" loop="true" volume="0.16" duration="36" role="emotional" perspective="editorial" anchor="那就从明天说起" edge="end" end_anchor="我听得见" fade="0.7" envelope="0:0.15,0.55:0.6,1:0.15">Sparse felt piano with a soft sustained cello, slow understated homecoming melody, spacious notes without a dramatic swell, no vocals</sound>
  <sequential>
    narrator: 许知夏把收摊的牌子翻了过去，[quiet, drawing closer] 陈伯却没坐下，只<soft>把纸慢慢摊平</soft>。
    陈伯: 这是林姨抄给我的，[trying to sound practical] <emphasis>明早七点二十</emphasis>，去南站的车。
    <parallel>
      <sequential>
        陈伯: 她要搬去女儿家，[careful, avoiding her eyes] 临走还问，那只钟<emphasis>修好了没有</emphasis>。
        陈伯: 你别笑话我，[quiet, admitting the truth] 那只钟<slow>早就修好了</slow>。
      </sequential>
      <parallel reaction="true">
        许知夏: 嗯，[gently encouraging him] 我听着呢。
      </parallel>
    </parallel>
    陈伯: 她每天来问，[regret, letting her understand] 我就每天说不急，想着明天，她总还会<soft>再来一趟</soft>。
    陈伯: 后来她问我，[vulnerable, recalling her question] 到了那边，能不能也找个人，<emphasis>陪她买菜</emphasis>。
    陈伯: 我说铺子走不开，[quiet, ashamed of retreating] 其实是<slow>我没敢答应</slow>。
    陈伯: 六十多年都在这条街上，[holding back, becoming honest] 我怕走了，连个<soft>认得我的人</soft>都没有。
    许知夏: 林姨不就认得您吗，[gentle, answering his fear] 您爱吃什么，哪条腿怕冷，她比我们<emphasis>都清楚</emphasis>。
    陈伯: 可我昨天还跟她说，[heavy, regretting his words] 钟修好了就拿走，<stress>不用再来了</stress>。
    陈伯: 我这张嘴，[small, asking for help] 一辈子会报时，怎么就说不出一句<soft>舍不得</soft>。[exhales]
    许知夏: 师父，[steady, offering a way] 昨天没说好，<emphasis>那就从明天说起</emphasis>。
    阿川: 电池装好了，[quiet, putting joking aside] 我把收音机搁窗台上，陪您<soft>听着</soft>。
    许知夏: 话筒给您，[warm, giving him space] 就当林姨还坐在柜台前，别想着全街的人。
    陈伯: 好，[gathering courage, to himself] 你们别关门，她要是路过，<soft>我听得见</soft>。
  </sequential>
</parallel>
<#0.7#>
<parallel stage="许知夏=-0.3, 阿川=0.4, 陈伯=0.1, 林姨=0.55" holds="留给我吧~1.0">
  <sound name="shop_rain" loop="true" volume="0.25" duration_class="texture" role="physical" perspective="in_world">Soft rain dripping from a shop awning just outside an open doorway, quiet distant street wash, no voices or machinery</sound>
  <sound name="broadcast_chime" volume="0.3" duration_class="reaction" role="transition" perspective="in_world" anchor="七点到了" edge="end" pan="0.4">Two rounded electronic interval-signal notes through a small old portable radio speaker, narrow bandwidth, soft static tail, no speech</sound>
  <sound name="basket_set_down" duration="2" role="physical" perspective="in_world" anchor="篮子轻轻碰了一下门槛" edge="start" pan="0.55">Wicker shopping basket softly touching a stone doorstep, handle creak and vegetable leaf rustle, close dry foreground</sound>
  <sound name="homeward_piano" loop="true" volume="0.16" duration="36" role="emotional" perspective="editorial" anchor="另一张在这儿" edge="end" end_anchor="一起带走" resume="true" fade="0.7" envelope="0:0.2,0.55:0.65,1:0.1">Sparse felt piano with a soft sustained cello, slow understated homecoming melody, spacious notes without a dramatic swell, no vocals</sound>
  <sound name="clock_restart" volume="0.65" duration="4" role="physical" perspective="in_world" anchor="钟摆轻轻一推" edge="end" pan="0.1">A small wooden mantel clock starting with one delicate mechanism click followed by steady gentle pendulum ticking, close dry detail, no chime</sound>
  <sequential>
    narrator: 七点到了，[hushed, expectant] 窗台上的收音机亮了，陈伯把话筒<soft>往身前挪了挪</soft>。
    陈伯: <radio_broadcast> 老街的街坊们，[careful, finding his courage] 今天这趟广播，我想找一位<soft>还没回家的人</soft>。
    陈伯: <radio_broadcast> 你那只钟修好了，[tender, speaking to her] 一分也不差，是我一直<emphasis>舍不得还</emphasis>。
    陈伯: <radio_broadcast> 明早七点二十，[steadier, making a promise] 我在巷口等你，铺子的钥匙，<soft>交给小许了</soft>。
    陈伯: <radio_broadcast> 就是车票我还没买，[vulnerable, asking her] 你旁边那个位置，<slow>留给我吧</slow>。
    narrator: 门外，[quiet, listening closely] 一只<emphasis>篮子轻轻碰了一下门槛</emphasis>。
    林姨: <far> 陈师傅，[brisk, hiding her relief] 你这广播全街都听见了，明天可不许<stress>又说不急</stress>。
    陈伯: 不说了，[eager, turning toward her] 我现在就去买票，你等我<fast>一小会儿</fast>！
    林姨: <near> 买什么，[tender beneath briskness] 我那天买的就是两张，<soft>另一张在这儿</soft>。
    陈伯: 你还真，[relieved, almost disbelieving] 给我留着啊。
    林姨: 你的脾气我不知道吗，[fond, letting him understand] 修个钟都要人等半个月，<soft>我就再等等</soft>。
    阿川: 陈伯，[soft, remembering his part] 那我明早来，帮您<emphasis>背箱子</emphasis>。
    许知夏: 店里有我，[warm, taking responsibility] 您记得把林姨的钟，<soft>一起带走</soft>。
    narrator: 陈伯取下那只钟，[warm, quietly releasing] 把<emphasis>钟摆轻轻一推</emphasis>。
    林姨: 这回，[softly, smiling to him] 走得正好。
    <#0.8#>
    陈伯: 嗯，[quiet, finally ready] 咱们也走吧。
  </sequential>
</parallel>"""
