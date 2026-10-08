"""
Canonical Sanskrit Corpus Story Documents (Trilingual: Sanskrit, English, Hindi)
Extracted faithfully from data/sanskrit_corpus.txt and data/Rag-docs.txt.
"""

from typing import Dict, Any

STORY_DOCUMENTS: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "key": "foolish_servant",
        "short_title": "1. मूर्खभृत्यः (Foolish Servant)",
        "title_sa": "मूर्खभृत्यस्य कथा",
        "title_en": "Story of the Foolish Servant (Shankhanada)",
        "title_hi": "मूर्ख नौकर शंखनाद की कथा",
        "genre": "Classical Sanskrit Didactic Folktale (नीतिकथा)",
        "source": "sanskrit_corpus.txt — Story 1",
        "default_query": "Why did the foolish servant ruin the sugar?",
        "summary": "Govardhanadas orders his foolish servant Shankhanada through a series of literal-minded blunders with sugar, a puppy, milk, and black soot, leading to the famous moral shloka on foolish associates.",
        "shloka_sa": "वरम् भृत्यविहिनस्य जिवितम् श्रमपूरितम् ।\nमूर्खभृत्यस्य संसर्गात् सर्वम् कार्यम् विनश्यति ॥",
        "shloka_en": "\"Better is a servant-less life filled with physical toil;\nFor through association with a foolish servant, every endeavor is completely ruined.\"",
        "shloka_hi": "\"नौकर के बिना मेहनत और परिश्रम से भरा जीवन जीना कहीं बेहतर है;\nक्योंकि एक मूर्ख नौकर की संगति से सारा का सारा काम पूरी तरह नष्ट हो जाता है।\"",
        "content_sa": """\"अरे शंखनाद, गच्छापणम्, शर्कराम् आनय ।\" इति स्वभृत्यम् शंखनादम् गोवर्धनदासः आदिशति ।

ततः शंखनादः आपणम् गच्छति, शर्कराम् जीर्णे वस्त्रे न्यस्यति च । तस्मात् जीर्णवस्त्रात् मार्गे एव सर्वापि शर्करा स्त्रवति ।

ततः गोवर्धनदासः कोपेन शंखनादम् वदति, \"अरे मूढ, कुत्रास्ति शर्करा ? शर्करादिकम् एवम् जीर्णेन वस्त्रेण न एवानयन्ति कदापि । इतःपरम् किमपि वस्तुजातम् दृढायाम् सन्चिकायाम् निक्षिप्य आनय च \" इति ।

अत्रान्तरे गोवर्धनदासस्य पुत्रः \"श्वानशावकम् आनय\" इति शंखनादम् आदिशति । आज्ञापालकः शंखनादः श्वानशावकम् सन्चिकायाम् क्षिपति, सन्चिकाम् वस्त्रेण आच्छादयति च । तेन शावकस्य श्वासः रुध्दः भवति । सः च श्वानशावकः पञ्चत्वम् गच्छति ।

तदा गोवर्धनदासः शंखनादम् अभिधावति क्रोधेनाक्रोशति च, \" धिक् मुढ, श्वानादिकम् दोरकेण बद्ध्वा आनयन्ति इति अपि नावगच्छसि किम् ?” इति ।

ततः कदाचित् गोवर्धनदासस्य भार्या \"दुग्धम् आनय\" इति तस्मै कथयति । सः आपणम् गच्छति पात्रे दुग्धम् आदाय दोरकेण बद्ध्वा कर्षति । मार्गे पात्रम् लुठति । पात्रात् दुग्धम् सर्वत्र प्रवहति ।

तेन हताशः गोवर्धनदासः तम् वदति \"भो महापंडित, अपसर; कृष्णम् भवतु ते मुखम् ” इति ।

तदा आज्ञापालकः शंखनादः बहिः गच्छति कज्जलेन मुखम् लिम्पति । तेन तस्य मुखम् कृष्णम् भवति । ततः कृष्णमुखः सः प्रत्यागच्छति ।

गोवर्धनदासः विरुपम् तम् पश्यति आत्मानम् एव निन्दति ललाटम् हस्तेन ताडयति वदति च —

वरम् भृत्यविहिनस्य जिवितम् श्रमपूरितम् ।
मूर्खभृत्यस्य संसर्गात् सर्वम् कार्यम् विनश्यति ॥""",
        "content_en": """\"Hey Shankhanada, go to the market and bring sugar!\" Thus orders Govardhanadas to his servant Shankhanada.

Then Shankhanada goes to the shop and puts the sugar into a torn piece of cloth. On the way back, all the sugar spills and leaks out from the torn cloth.

Govardhanadas angrily scolds Shankhanada: \"Hey fool, where is the sugar? One never brings sugar and groceries wrapped in a torn cloth! From now on, whatever item you bring, put it in a sturdy bag and bring it!\"

Meanwhile, Govardhanadas's son orders Shankhanada: \"Bring the puppy!\" The obedient Shankhanada puts the puppy inside a bag and covers the bag tightly with a cloth. Due to this, the puppy's breathing is choked, and the puppy passes away.

Govardhanadas runs after Shankhanada and shouts in rage: \"Fie on you, idiot! Don't you even understand that animals like dogs are brought by tying them with a rope?\"

Later, Govardhanadas's wife tells him: \"Bring milk.\" He goes to the shop, gets milk in a pot, ties the pot with a rope, and drags it along the street. On the road, the pot overturns and rolls, and the milk spills everywhere.

Utterly frustrated and defeated, Govardhanadas tells him: \"O great scholar, get away from here; may your face turn black!\"

The literal-minded Shankhanada goes outside and smears his face with black soot (kajjal). Thus his face turns pitch black. Then he returns with his blackened face.

Govardhanadas sees his ridiculous, disfigured face, curses his own fate, strikes his forehead in despair, and laments:

\"Better is a servant-less life filled with constant physical labor;
For through association with a foolish servant, every endeavor is completely ruined.\"""",
        "content_hi": """\"अरे शंखनाद, बाजार जाओ और शक्कर (चीनी) लेकर आओ!\" इस प्रकार गोवर्धनदास ने अपने नौकर शंखनाद को आदेश दिया।

तब शंखनाद बाजार जाता है और शक्कर को एक फटे-पुराने कपड़े में बांध लेता है। उस फटे कपड़े के छेदों से रास्ते में ही सारी शक्कर गिरकर बह जाती है।

तब गोवर्धनदास क्रोधित होकर शंखनाद से कहता है: \"अरे मूर्ख, शक्कर कहाँ है? शक्कर जैसी चीजें कभी भी फटे कपड़े में नहीं लाई जातीं! अब से जो भी सामान लाओ, उसे किसी मजबूत थैली (बोरी) में रखकर लाया करो।\"

इसी बीच गोवर्धनदास का पुत्र शंखनाद को आदेश देता है: \"कुत्ते के पिल्ले को लेकर आओ!\" आज्ञाकारी शंखनाद उस पिल्ले को थैली में डाल देता है और थैली को कपड़े से कसकर बांध देता है। इससे पिल्ले का दम घुट जाता है और वह मर जाता है।

तब गोवर्धनदास शंखनाद की ओर दौड़ता है और गुस्से से चिल्लाता है: \"धिक्कार है तुझ मूर्ख पर! क्या तुझे इतना भी नहीं पता कि कुत्ते आदि जीवों को रस्सी से बांधकर लाया जाता है?\"

इसके बाद एक दिन गोवर्धनदास की पत्नी उससे कहती है: \"दूध लेकर आओ।\" वह बाजार जाता है, बर्तन में दूध लेता है और बर्तन को रस्सी से बांधकर जमीन पर घसीटता हुआ लाता है। रास्ते में बर्तन लुढ़क जाता है और सारा दूध जमीन पर बह जाता है।

इससे हताश और निराश होकर गोवर्धनदास उससे कहता है: \"अरे महापंडित, मेरी आँखों के सामने से हट जा; तेरा मुँह काला हो जाए!\"

तब आज्ञाकारी शंखनाद बाहर जाता है और काजल से अपना पूरा मुँह पोत लेता है, जिससे उसका चेहरा बिल्कुल काला हो जाता है। फिर वह काले मुँह के साथ वापस आता है।

गोवर्धनदास उसके इस बिगड़े हुए रूप को देखकर स्वयं को कोसता है, अपना माथा पीटता है और कहता है —

\"नौकर के बिना मेहनत और परिश्रम से भरा जीवन जीना कहीं बेहतर है;
क्योंकि एक मूर्ख नौकर की संगति से सारा का सारा काम पूरी तरह नष्ट हो जाता है।\""""
    },

    2: {
        "id": 2,
        "key": "clever_kalidasa",
        "short_title": "2. कालीदासस्य (Clever Kalidasa)",
        "title_sa": "चतुरस्य कालीदासस्य कथा",
        "title_en": "Story of Clever Kalidasa & King Bhoja's 99-Crore Gem Riddle",
        "title_hi": "चतुर कालिदास और राजा भोज की 99 करोड़ रत्नों की कथा",
        "genre": "Classical Royal Court Lore & Subhashita (सभा-चातुर्यम्)",
        "source": "sanskrit_corpus.txt — Story 2",
        "default_query": "What did King Bhoja announce in his court?",
        "summary": "King Bhoja proclaims a 1-lakh rupee prize for any new poem, which court scholars with photographic memories repeatedly sabotage, until Kalidasa crafts an inescapable 99-crore gem debt riddle.",
        "shloka_sa": "स्वस्ति श्री भोजराजन् त्वमखिलभुवने धार्मिकः सत्यवक्ता ।\nपित्रा ते संगृहीता नवनवतिमिता रत्नकोट्यो मदीयः ।\nतांस्त्वं देहीति राजन् सकलबुधजनैर्जायते सत्यं एतद् ।\nनो वा जानन्ति यत्तन्ममकृतिमपि नो देहि लक्षं ततो मे ॥",
        "shloka_en": "\"May there be auspiciousness, O King Bhoja! In all the worlds, you are righteous and a speaker of truth.\nNinety-nine crore precious stones belonging to me were borrowed by your father.\nReturn them to me, O King, for all the scholars in your court know that this is true!\nOr if they do not know it, then neither did they know my poem—so grant me my one lakh rupees!\"",
        "shloka_hi": "\"हे राजा भोज, आपका कल्याण हो! आप पूरे संसार में परम धार्मिक और सत्यवादी हैं।\nआपके स्वर्गीय पिताश्री ने मेरे 99 करोड़ बहुमूल्य रत्न उधार लिए थे।\nहे राजन्, वे रत्न मुझे लौटा दीजिए, क्योंकि आपके दरबार के ये सभी विद्वान इस सत्य को भली-भांति जानते हैं!\nऔर यदि ये विद्वान इसे नहीं जानते, तो इसका अर्थ है कि वे मेरी इस कविता को भी नहीं जानते थे—इसलिए मुझे मेरा एक लाख रुपये का पुरस्कार दीजिए!\"",
        "content_sa": """घोषितं कदाचित् भोजराज्ञा, यदि कोऽपि कविः मम दरबारे नूतनं काव्यं पठति तर्हि ददामि तस्मै लक्षरुप्यकाणि इति ।
श्रुत्वा एतद्, बहवः खलु आगताः कवयः दरबारे स्वकाव्यपठनं कर्तुं ।

परन्तु दरबारे अविद्यन्त केचन विद्वानाः ये कं अपि काव्यं प्रथमश्रुत्यनन्तरं एव सम्पूर्णतया पुनरोक्तुं शक्ताः । इतरे च द्वीतीयपठनानन्तरं पुनरुक्तवन्ताः । अन्ये तृतीयपठनात् पुनरुक्तवन्ताः ।

अतः यदा कोऽपि कविः आगतवान्, स्वस्य नवकाव्यं च पठितवान्, केचन विद्वानाः अकथयन् न खलु नूतनं एतद् । पश्य, वयमपि एनं जानीमहे । ततः पुनरवदन् तद काव्यं । तदनन्तरं ये द्विपाठिनः अभवन् तेऽपि तदेव अकथयन् अवदन् च काव्यं । तदनन्तरं त्रिपाठिनः तथैव अकुर्वन् । अतः न कोऽपि कविः प्राप्नोतुं अशक्नोत् लक्षरुप्यकाणि ।

कालीदासाय न खलु अरुच्यत एतद् । कंचित् नवकविं सः स्वकक्षे आनीतवान् तं एतद् सुभाषितं च दत्तवान् —

स्वस्ति श्री भोजराजन् त्वमखिलभुवने धार्मिकः सत्यवक्ता ।
पित्रा ते संगृहीता नवनवतिमिता रत्नकोट्यो मदीयः ।
तांस्त्वं देहीति राजन् सकलबुधजनैर्जायते सत्यं एतद् ।
नो वा जानन्ति यत्तन्ममकृतिमपि नो देहि लक्षं ततो मे ॥

दरबारे कविः तत् काव्यं अपठत् । न खलु वक्तुं अशक्नुवन् केऽपि विद्वानाः यत् जानन्ति तत् काव्यं इति ।
अतः अप्राप्नोत् कविः लक्षरुप्यकाणि । चतुरः खलु कालीदासः !""",
        "content_en": """King Bhoja once proclaimed: \"If any poet recites a brand new poem in my royal court, I shall award him one lakh rupees (100,000 rupees)!\"
Hearing this proclamation, many poets thronged the royal assembly to recite their new compositions.

However, in the royal court, there were certain scholars who possessed photographic memories:
- Some could re-recite an entire poem completely after hearing it just once (Ekapathins).
- Others could recite it after hearing it twice (Dvipathins).
- Yet others could recite it after hearing it thrice (Tripathins).

Therefore, whenever any poet presented a fresh poem, the one-hearing scholars immediately interjected: \"This poem is not new at all! Look, we already know it by heart!\" and recited it word-for-word. Then the two-hearing scholars repeated it, followed by the three-hearing scholars. Consequently, no poet was ever able to claim the one lakh rupees prize!

Kalidasa strongly disapproved of this unfair trick played on poor poets. He brought a discouraged new poet to his private chamber and gave him this ingenious verse:

\"May there be auspiciousness, O King Bhoja! In all the worlds, you are righteous and a speaker of truth.
Ninety-nine crore (990,000,000) precious gems belonging to me were borrowed by your father.
Return them to me, O King, for all the learned scholars in your royal court know this to be true!
Or if they do not know it, then neither did they know my poem—so grant me my one lakh rupees reward!\"

The poet presented this poem before the royal court. The court scholars were completely checkmated: if they claimed they already knew the poem, they would be legally certifying under oath that King Bhoja owed the poet 99 crore precious gems! If they admitted they did not know it, the poem was officially proven brand new.

None of the scholars dared say they knew the poem. Consequently, the poet won the one lakh rupees reward. Clever indeed was Kalidasa!""",
        "content_hi": """एक समय राजा भोज ने अपने दरबार में घोषणा की: \"यदि कोई कवि मेरे दरबार में कोई बिल्कुल नया काव्य सुनाएगा, तो मैं उसे एक लाख रुपये का पुरस्कार दूँगा।\"
यह सुनकर दूर-दूर से बहुत से कवि अपनी नई रचनाएं सुनाने के लिए राजदरबार में उपस्थित हुए।

परंतु राजा भोज के दरबार में असाधारण स्मरण-शक्ति वाले कुछ विद्वान रहते थे:
- कुछ विद्वान ऐसे थे जो किसी भी काव्य को केवल एक बार सुनकर ही पूरा का पूरा दोहरा देते थे (एकपाठी)।
- कुछ विद्वान दो बार सुनकर कंठस्थ कर लेते थे (द्विपाठी)।
- और अन्य विद्वान तीन बार सुनकर पूरी कविता सुना देते थे (त्रिपाठी)।

अतः जब भी कोई कवि आता और अपनी नई कविता सुनाता, तो एकपाठी विद्वान तुरंत कह देते: \"यह कोई नई रचना नहीं है! देखो, हम तो इसे पहले से जानते हैं,\" और वे उसे ज्यों का त्यों सुना देते। फिर द्विपाठी और त्रिपाठी विद्वान भी उसे दोहरा देते। इस चाल के कारण कोई भी कवि एक लाख रुपये का पुरस्कार नहीं जीत पाता था।

महाकवि कालिदास को कवियों के साथ यह अन्याय बिल्कुल अच्छा नहीं लगा। उन्होंने एक नए और निराश कवि को अपने कक्ष में बुलाया और उसे यह चमत्कारी श्लोक सिखाया —

\"हे राजा भोज, आपका कल्याण हो! आप सम्पूर्ण संसार में परम धार्मिक और सत्यवादी राजा हैं।
आपके स्वर्गीय पिताश्री ने मेरे 99 करोड़ बहुमूल्य रत्न उधार लिए थे।
हे राजन्, वे रत्न मुझे वापस लौटा दीजिए, क्योंकि आपके दरबार के ये सभी विद्वान इस बात को भली-भांति जानते हैं!
और यदि ये विद्वान इसे नहीं जानते, तो इसका अर्थ है कि वे मेरी इस कविता को भी नहीं जानते थे—इसलिए मुझे मेरा एक लाख रुपये का पुरस्कार दीजिए!\"

दरबार में कवि ने जाकर यह श्लोक पढ़ा। अब दरबारी विद्वान भारी धर्मसंकट में फंस गए: यदि वे कहते कि वे यह रचना पहले से जानते हैं, तो राजा को 99 करोड़ रत्न चुकाने पड़ते; और यदि वे कहते कि वे नहीं जानते, तो कविता अपने आप नई सिद्ध हो जाती!

दरबार का कोई भी विद्वान यह कहने का साहस नहीं कर सका कि वे इस काव्य को जानते हैं। इस प्रकार उस कवि को एक लाख रुपये का पुरस्कार प्राप्त हुआ। सचमुच, कालिदास अत्यंत चतुर थे!"""
    },

    3: {
        "id": 3,
        "key": "old_woman_bell",
        "short_title": "3. वृद्धायाः चातुर्यम् (Old Woman & Bell)",
        "title_sa": "वृद्धायाः चातुर्यम्",
        "title_en": "The Old Woman's Cleverness & the Ghantakarna Bell Demon",
        "title_hi": "बुढ़िया की चतुराई और घण्टाकर्ण राक्षस की कथा",
        "genre": "Classical Sanskrit Folktale on Wit vs. Superstition (चातुर्य-कथा)",
        "source": "sanskrit_corpus.txt — Story 3",
        "default_query": "Who was Ghantakarna demon and why was the bell ringing?",
        "summary": "Panic strikes Chitrapuram as citizens believe a man-eating demon Ghantakarna is ringing a bell on the peak, until a shrewd elderly woman uncovers the truth of playful monkeys and captures the bell using sweet fruits.",
        "shloka_sa": "बुद्धिर्यस्य बलं तस्य निर्बुद्धेस्तु कुतो बलम् ।\nवने सिंहो मदोन्मत्तः शशकेन निपातितः ॥",
        "shloka_en": "\"He who possesses intellect possesses true strength; how can the intellect-less have strength?\nEven an intoxicated lion in the forest was brought down by a tiny hare.\"",
        "shloka_hi": "\"जिसके पास बुद्धि है, उसी के पास वास्तविक बल है; बुद्धिहीन के पास बल कहाँ?\nजंगल में मदमस्त शेर को भी एक छोटे से खरगोश ने मार गिराया था।\"",
        "content_sa": """आसीत् चित्रपुरम् नाम किमपि नगरं श्रीपर्वतस्य समीपे । \"पर्वतस्य शिखरप्रदेशे घण्टाकर्णः नाम राक्षसः प्रतिवसती\" ति जनप्रवादः अवर्तत् ।

अथैकदा कश्चन चोरः घण्टामेकां चोरयित्वा वनं गतः, व्याघ्रेण च हतः । तदा सा घण्टा वने एव अपतत् ।

अन्यस्मिन् दिने केचन वानराः तत्र आगच्छन् । कुतूहलेन तां घण्टां हस्ते धृत्वा अधुन्वन् । अकस्मादेव घण्टानादः अजायत् ।
घण्टानादेन चकिताः ते पुनः पुनः घण्टामधुन्वन्, घण्टानादं च अकुर्वन् ।

चित्रपुरस्थाः नागरिकाः वारं वारं पर्वतशिखरप्रदेशात् घण्टानादमाकर्णयन् । भयाकुलाः ते अचिन्तयन् \"नूनं शिखरप्रदेशे घण्टकर्णः नाम राक्षसः वर्तते, यः मनुष्यान् खादति, घण्टां च वादयति\" ।

एवं च भीत्या पौरजनाः अन्यत्र गन्तुं प्रारभन्त ।

तदा चिन्ताकुलः नृपः उदघोषयत् \"यः घण्टकर्णं नाशयेत्, तस्मै विपुलं सुवर्णं यच्छेयम् अहम्\" इति ।

तत् श्रुत्वा काचन वृद्धा वनं गता । तत्र च कंचित् कालं निभृतम् अतिष्ठत् ।
'वानराः एव घण्टां वादयन्ति' इति सा अपश्यत् ।

अन्येद्युः सा वानरेभ्यः मधुराणि फलानि अयच्छत् । यावत् ते फलभक्षणमग्नाः संजाताः, तावदेव तां घण्टामादाय प्रमुदिता सा नृपं प्रत्यागच्छत्, अवदच्च \"राजन्, घण्टाकर्णः हतः मया\" इति ।

नृपः तस्यै प्रभूतं सुवर्णमयच्छत् । ततः प्रभृति पौरजनाः घण्टारवं न आकर्णयन् । भयमुक्ताः ते नगरं प्रतिन्यवर्तन्त ।""",
        "content_en": """Near the sacred mountain Shri Parvata, there existed a flourishing city named Chitrapuram. A terrifying rumor spread among the populace: \"Upon the mountain peak dwells a fierce demon named Ghantakarna!\"

Once, a thief stole a bell, ran into the dense forest to escape, and was slain by a tiger. The bell was left abandoned on the forest floor.

On another day, a troop of monkeys happened upon the spot. Driven by curiosity, they picked up the bell and shook it vigorously. Suddenly, a resonant chime rang out!
Delighted and intrigued by the metallic ringing, the monkeys repeatedly shook the bell, causing continuous chimes to echo through the hills.

The citizens of Chitrapuram repeatedly heard the eerie ringing echoing from the mountain summit. Stricken with terror, they concluded: \"Surely the demon Ghantakarna dwells upon the peak—he devours humans and tolls his bell!\"

Driven by sheer panic, the citizens began abandoning the city to seek refuge elsewhere.

Alarmed by the desertion of his capital, the king issued a royal decree: \"Whoever destroys Ghantakarna shall be rewarded with heaps of pure gold!\"

Hearing this, a wise old woman ventured into the forest. She quietly hid behind trees and observed the summit.
She discovered the reality: it was merely curious monkeys ringing the abandoned bell!

The next day, she returned carrying a basket of sweet, ripe fruits and scattered them before the monkeys. While the monkeys eagerly feasted on the delicious fruits, she quietly picked up the bell and joyfully brought it before the king, announcing: \"O King, I have vanquished Ghantakarna!\"

The overjoyed king rewarded her with abundant gold. From that day onward, no eerie bells were heard, and free from superstition and fear, all the citizens happily returned to their homes.""",
        "content_hi": """श्रीपर्वत के निकट 'चित्रपुर' नाम का एक सुंदर नगर था। वहाँ के लोगों में एक भयानक अफवाह फैल गई: \"पर्वत की चोटी पर 'घण्टाकर्ण' नाम का एक राक्षस रहता है!\"

एक बार एक चोर कहीं से एक घंटी चुराकर घने जंगल की ओर भागा, जहाँ एक बाघ ने उसे मार डाला। वह घंटी जंगल में ही जमीन पर गिर गई।

दूसरे दिन कुछ बंदर वहाँ आए। उत्सुकतावश उन्होंने उस घंटी को हाथ में लेकर हिलाया, तो अचानक घंटी की टन-टन आवाज गूँज उठी।
घंटी की आवाज से रोमांचित होकर बंदर उसे बार-बार हिलाने और जोर-जोर से बजाने लगे।

चित्रपुर के नगरवासियों ने पहाड़ की चोटी से बार-बार घंटी की आवाज सुनी। डर के मारे कांपते हुए वे सोचने लगे: \"निश्चय ही चोटी पर घण्टाकर्ण राक्षस रहता है, जो मनुष्यों को खा जाता है और घंटी बजाता है!\"

इस भयानक डर के कारण नगरवासी अपना घर-बार छोड़कर दूसरी जगह भागने लगे।

नगर को खाली होते देख चिंतित राजा ने घोषणा करवाई: \"जो कोई भी घण्टाकर्ण राक्षस का नाश करेगा, उसे मैं प्रचुर मात्रा में सोना (स्वर्ण) पुरस्कार में दूँगा।\"

यह सुनकर एक चतुर और समझदार बुढ़िया जंगल में गई। वह कुछ समय तक चुपचाप छिपकर देखती रही।
उसने अपनी आँखों से देख लिया कि कोई राक्षस नहीं है, बल्कि बंदर ही खेल-खेल में घंटी बजा रहे हैं!

अगले दिन वह बुढ़िया बहुत सारे मीठे-मीठे फल लेकर जंगल गई और बंदरों के सामने बिखेर दिए। जब बंदर फल खाने में पूरी तरह मग्न हो गए, तो बुढ़िया ने चुपके से घंटी उठा ली और प्रसन्न होकर राजा के पास पहुँची। उसने राजा से कहा: \"महाराज! मैंने घण्टाकर्ण राक्षस को समाप्त कर दिया है।\"

प्रसन्न होकर राजा ने बुढ़िया को ढेर सारा सोना पुरस्कार में दिया। उसके बाद नगरवासियों को कभी घंटी की आवाज सुनाई नहीं दी और अंधविश्वास व भय से मुक्त होकर सभी नागरिक अपने नगर वापस लौट आए।"""
    },

    4: {
        "id": 4,
        "key": "devotee_in_flood",
        "short_title": "4. देवभक्तस्य कथा (Devotee in Flood)",
        "title_sa": "देवभक्तस्य कथा",
        "title_en": "Story of the Devotee in the Flood & the 6 Virtues of Human Effort",
        "title_hi": "बाढ़ में ईश्वरभक्त और पुरुषार्थ के छह सद्गुणों की कथा",
        "genre": "Classical Sanskrit Parable on Effort vs. Fatalism (उद्यम-कथा)",
        "source": "sanskrit_corpus.txt — Story 4",
        "default_query": "Why did the devotee drown in water and what did God say?",
        "summary": "An idle devotee trusts God to rescue him without lifting a finger, refusing 3 human rescuers during a flood until he drowns, learning in heaven that God acts through human initiative and the 6 virtues.",
        "shloka_sa": "उद्यमः साहसम् धैर्यम् बुद्धिः शक्तिः पराक्रमः ।\nषडेते यत्र वर्तन्ते तत्र देवः साहाय्यकृत् ॥",
        "shloka_en": "\"Industry (effort), courage, patience, intellect, strength, and valor —\nWherever these six virtues reside, there alone does God extend His divine aid.\"",
        "shloka_hi": "\"परिश्रम, साहस, धैर्य, बुद्धि, शक्ति और पराक्रम —\nये छह गुण जहाँ विद्यमान होते हैं, वहाँ भगवान भी अवश्य सहायता करते हैं।\"",
        "content_sa": """एकः परमः देवभक्तः अस्ति । सः प्रतिदिने भक्त्या देवस्य प्रार्थनाम् करोति —
\"देव, कृपया मह्यं आरोग्यम् ददातु, धनम् ददातु\" इति । सः किंचित् अपि प्रयत्नम् न करोति, कार्यम् न करोति ।
देवः साहाय्यम् करिष्यति इति तस्य विश्वासः ।

एकस्मिन् दिने सः वृषभशकटे गच्छति स्म । मृण्मार्गः आसीत् । तदा वृष्टेः आरम्भः अभवत् । घटिकात्रयं वृष्टिः निरंतरम् आगतवती ।
तदा तस्य शकटस्य चक्रम् मार्गे अन्तः गतम् । सः उपविश्य \"देव, कृपया साहाय्यम् करोतु, अहम् परमभक्तः अस्मि । कृपया साहाय्यम् करोतु\" इति प्रार्थनाम् कृतवान् ।

तदा मार्गे एकः सज्जनः आगतवान् । भक्तम् दृष्ट्वा पृष्टवान् \"भो मित्र, किंचित् साहाय्यम् आवश्यकम् वा ?\" इति ।
तदा भक्तः उक्तवान् \"भवतः साहाय्यम् न आवश्यकम् । देवः अस्ति । सः एव साहाय्यम् करोति\" इति । तदा सज्जनः गतवान् ।

किंचित् समयानंतरम्, अन्यः एकः आगतवान् । सः अपि तदैव पृष्टवान् \"भो मित्र, किंचित् साहाय्यम् आवश्यकम् वा ?\" इति ।
तदा भक्तः उक्तवान् \"भवतः साहाय्यम् न आवश्यकम् । देवः अस्ति । सः एव साहाय्यम् करोति\" इति । सः अपि गतवान् ।

इदानीं वृष्टिः अधिका अभवत् । जलम् तस्य कण्ठपर्यंतम् आगतम् ।
तदा पुनः एकः आगतवान्, पृष्टवान् च \"भो मित्र, किंचित् साहाय्यम् आवश्यकम् वा ?\" इति । तदा भक्तः पुनः उक्तवान् \"भवतः साहाय्यम् न आवश्यकम् । देवः अस्ति । सः एव साहाय्यम् करोति\" इति । सः अपि गतवान् ।

वृष्टिः अधिका अभवत् । सः जले मृतवान् ।

तदा सः स्वर्गम् गतवान् । सः देवम् पृष्टवान् \"देव, अहम् भवतः परमभक्तः । यदा मम कष्टः अभवत्, भवान् किंचित् अपि साहाय्यम् न कृतवान् । भवान् द्रष्टुम् अपि न आगतवान् । किमर्थम् ?\"

तदा देवः उक्तवान् \"भो भक्त, अहम् त्रिवारम् आगतवान् । पृष्टवान् च — 'साहाय्यम् आवश्यकम् वा ?' इति । परंतु भवान् न स्वीकृतवान् । यदि भवान् प्रयत्नम् एव न करोति चेत्, अहम् कथम् साहाय्यम् करोमि ?\"

तदा भक्तस्य ज्ञानोदयः अभवत् । यदि वयम् प्रयत्नम् कुर्मः, तर्हि एव देवः साहाय्यम् करोति —

उद्यमः साहसम् धैर्यम् बुद्धिः शक्तिः पराक्रमः ।
षडेते यत्र वर्तन्ते तत्र देवः साहाय्यकृत् ॥""",
        "content_en": """There was an ardent devotee of God. Every single day, he prayed with utmost devotion:
\"O Lord, please bestow good health upon me and grant me abundant wealth!\"
Yet, he never made the slightest personal effort nor performed any honest work. He blindly believed: \"God Himself will do everything for me.\"

One day, he was traveling in an ox-drawn cart along a dirt road. Heavy monsoon rains set in and poured without ceasing for three continuous hours.
The heavy wheel of the cart sank deep into the thick mud. Instead of getting down to push the wheel, the devotee sat motionless and pleaded: \"O God, please help me! I am your supreme devotee, please come and save me!\"

A kind passerby arrived, noticed his predicament, and asked: \"My friend, do you need some help?\"
The devotee replied: \"Your help is not needed. God is there; He alone will come and help me.\" The kind man went on his way.

After some time, a second person came by and asked the same question: \"Friend, do you require assistance?\"
The devotee repeated: \"Your help is not needed. God is there; He alone will help me.\" The second person departed.

Now the flooding intensified, and water rose all the way to his neck.
A third person rushed over and urgently called out: \"Friend, take my hand, do you need help?\" The stubborn devotee replied once more: \"Your help is not needed. God alone will save me.\" That rescuer left as well.

The floodwaters rose higher, and the devotee drowned in the deluge.

Reaching heaven, he bitterly confronted God: \"Lord! I was your supreme devotee! When I was drowning in misery, You did not help me at all. You did not even come to see me. Why?\"

God replied with compassion: \"O devotee, I came to you three separate times through those travelers and asked: 'Do you need help?' But you stubbornly refused each time. If you refuse to make any effort yourself, how can I help you?\"

At that moment, true wisdom dawned upon the devotee: God aids only those who exert personal initiative and labor —

\"Industry (perseverance), courage, patience, intellect, strength, and valor —
Wherever these six virtues reside, there alone does God extend His divine aid.\"""",
        "content_hi": """एक बहुत बड़ा ईश्वरभक्त था। वह प्रतिदिन बड़ी श्रद्धा और निष्ठा से भगवान की प्रार्थना करता था —
\"हे प्रभु, कृपया मुझे उत्तम स्वास्थ्य दीजिए और धन-संपत्ति दीजिए!\" परंतु वह स्वयं रत्ती भर भी प्रयास या परिश्रम नहीं करता था। उसका अंधविश्वास था कि भगवान स्वयं सब कुछ करेंगे।

एक दिन वह बैलगाड़ी से कहीं जा रहा था। रास्ता कच्ची मिट्टी का था। तभी मूसलाधार वर्षा शुरू हो गई और लगातार तीन घंटे तक बरसती रही।
बैलगाड़ी का पहिया कीचड़ में गहरा धंस गया। वह गाड़ी से उतरकर पहिया निकालने का प्रयास करने के बजाय वहीं बैठकर प्रार्थना करने लगा: \"हे भगवान, रक्षा कीजिए! मैं आपका परम भक्त हूँ, कृपया मेरी सहायता कीजिए!\"

तभी रास्ते से एक भला आदमी गुजरा। उसने भक्त को देखकर पूछा: \"अरे मित्र, क्या किसी सहायता की आवश्यकता है?\"
भक्त ने घमंड से कहा: \"मुझे तुम्हारी सहायता की कोई जरूरत नहीं है। भगवान हैं, वे ही मेरी सहायता करेंगे।\" वह भला आदमी चला गया।

कुछ देर बाद दूसरा व्यक्ति आया। उसने भी पूछा: \"मित्र, क्या तुम्हें किसी मदद की जरूरत है?\"
भक्त ने फिर वही उत्तर दिया: \"तुम्हारी मदद की कोई आवश्यकता नहीं है। भगवान स्वयं मेरी रक्षा करेंगे।\" वह भी चला गया।

अब बारिश और प्रचंड हो गई और बाढ़ का पानी उसके गले तक पहुँच गया।
तभी तीसरा व्यक्ति दौड़ता हुआ आया और बोला: \"मित्र, मेरा हाथ पकड़ो, क्या सहायता चाहिए?\" भक्त ने पुनः वही हठ दिखाया: \"मुझे तुम्हारी सहायता नहीं चाहिए, ईश्वर ही मेरी रक्षा करेंगे।\" वह भी चला गया।

पानी और बढ़ गया और भक्त पानी में डूबकर मर गया।

स्वर्ग पहुँचकर उसने व्यथित होकर भगवान से शिकायत की: \"हे प्रभु, मैं आपका परम भक्त था। जब मैं जीवन-मरण के संकट में था, तो आपने मेरी रत्ती भर भी सहायता नहीं की! आप मुझे देखने तक नहीं आए, ऐसा क्यों?\"

तब भगवान ने मुस्कुराते हुए समझाया: \"अरे मूर्ख भक्त! मैं ही उन तीन राहगीरों के रूप में तीन बार तुम्हारे पास आया था और पूछा था कि मदद चाहिए क्या? परंतु तुमने हर बार मेरा हाथ ठुकरा दिया। जब तुम स्वयं हाथ-पैर हिलाने का कोई पुरुषार्थ ही नहीं करोगे, तो मैं तुम्हारी सहायता कैसे कर सकता हूँ?\"

तब उस भक्त को सच्चा ज्ञान प्राप्त हुआ कि ईश्वर भी केवल उन्हीं की सहायता करते हैं जो स्वयं परिश्रम और उद्यम करते हैं —

\"उद्यम (परिश्रम), साहस, धैर्य, बुद्धि, शक्ति और पराक्रम —
ये छह गुण जहाँ विद्यमान होते हैं, वहाँ भगवान भी अवश्य सहायता करते हैं।\""""
    },

    5: {
        "id": 5,
        "key": "winter_grammar_riddle",
        "short_title": "5. शीतं बहु बाधति (Winter Riddle)",
        "title_sa": "शीतं बहु बाधति",
        "title_en": "The Winter Grammar Riddle (\"Sheetam Bahu Badhati\") & Kalidasa's Retort",
        "title_hi": "'शीतं बहु बाधति' की व्याकरण पहेली और कालिदास का प्रत्युत्तर",
        "genre": "Classical Paninian Sanskrit Grammatical Legend (व्याकरण-चातुर्यम्)",
        "source": "sanskrit_corpus.txt — Story 5",
        "default_query": "Why was badhati incorrect in sheetam bahu badhati?",
        "summary": "A haughty foreign scholar arrives to challenge King Bhoja's court, but when he utters an ungrammatical phrase 'sheetam bahu badhati' to his palanquin carrier, the disguised Kalidasa delivers a devastating grammatical correction.",
        "shloka_sa": "न तथा बाधते शीतं यथा बाधति बाधते ।",
        "shloka_en": "\"The winter cold does not afflict/hurt as much as your ungrammatical blunder 'badhati' hurts me!\"",
        "shloka_hi": "\"मुझे यह कड़ाके की सर्दी उतना कष्ट नहीं दे रही, जितना आपका अशुद्ध व्याकरण शब्द 'बाधति' कष्ट दे रहा है!\"",
        "content_sa": """सर्वे जानन्ति यत् भोजराज्ञः दरबारे अविद्यत कविः कालीदासः ।
कदाचित् एकः परदेशीयः पण्डितः भोजराज्ञे सन्देशं प्रक्षिप्तवान् । सन्देशे लिखितं, \"आगमिष्यामि अमुकदिवसे भवतः दरबारस्य पण्डितैः सह चर्चां विवादं च कर्तुम्\" इति । तथा भोजराजा दरबारे अकथयत् एषः पण्डितः आगमिष्यति इति ।

यस्मिन् दिवसे पण्डितः आगच्छति, तस्मिन् कालीदासः पालखीधारकस्य रूपं परिदधानः तस्य स्वागताय उपस्थितः भवति । न खलु जानाति पण्डितः यत् कालीदासः एव सः ।
पालखीं स्कन्धयोः वहन् निर्गतः कालीदासः पण्डितेन सह । तस्मिन् काले शिशिरः भवति ऋतुः, शीतः च पवनः देहं ताडयति इव ।
वदति पण्डितः, \"शीतं बहु बाधति\" इति ।

चतुरः कालीदासः त्वरया एव प्रतिवदति, \"न तथा बाधते शीतं यथा बाधति बाधते\" ।

आत्मनेपदी खलु 'बाध्' धातुः इति न विज्ञातं पण्डितेन ।
मन्यते सः, यदि एतस्मिन् राज्ये पालखीधारकाः अपि एतावत् जानन्ति संस्कृतं, तर्हि पण्डितैः सह मेलः मम पराभवाय एव ।
तथा कालीदासं आज्ञापयति न खलु इच्छामि एतस्मिन् राज्ये गन्तुम् । गृहे गमिष्यामः इति ।""",
        "content_en": """Everyone knows that the legendary poet Kalidasa was the jewel of King Bhoja's royal assembly.
Once, a haughty foreign scholar sent a formal challenge letter to King Bhoja. The letter announced: \"I shall arrive on such-and-such date to debate and discuss with the scholars of your court.\" King Bhoja announced in his royal court that this challenging scholar was arriving.

On the designated day when the scholar arrived, Kalidasa disguised himself in the modest garments of a palanquin bearer and went to receive him. The visiting scholar had no inkling that the bearer carrying his palanquin was none other than the master poet Kalidasa!

Hoisting the palanquin upon his shoulders, Kalidasa set off carrying the scholar. It was the peak of winter (Shishira ritu), and biting icy winds lashed mercilessly against the travelers' bodies.
Shivering from the cold, the scholar remarked in Sanskrit:
\"Sheetam bahu badhati\" (\"The cold hurts/afflicts very much\").

Instantly, clever Kalidasa retorted in verse:
\"Na tatha badhate sheetam yatha badhati badhate!\"
(\"The winter cold does not afflict/hurt as much as your ungrammatical blunder 'badhati' hurts me!\")

The scholar had failed to recognize that the Sanskrit verbal root 'baadh' (बाध् - to afflict/hurt) belongs strictly to the Atmanepada class (properly conjugated as 'badhate'), and never Parasmaipada ('badhati').
The scholar was seized with panic and thought: \"If even the humble palanquin bearers in this kingdom possess such flawless, instinctive command over Paninian Sanskrit grammar, confronting the royal assembly pandits will certainly lead to my utter humiliation and defeat!\"

He immediately commanded his disguised carrier: \"I have no desire to proceed into this kingdom; turn the palanquin around, we are returning straight home!\"""",
        "content_hi": """सभी जानते हैं कि राजा भोज के दरबार में महाकवि कालिदास विद्यमान थे।
एक बार किसी दूसरे देश के एक अहंकारी पंडित ने राजा भोज को संदेश भेजा। संदेश में लिखा था: \"मैं अमुक दिन आपके दरबार के विद्वानों के साथ शास्त्रार्थ और वाद-विवाद करने आऊँगा।\" राजा भोज ने अपने दरबार में घोषणा की कि यह विदेशी पंडित शास्त्रार्थ के लिए आने वाला है।

जिस दिन वह पंडित राज्य में पहुँचा, कालिदास एक साधारण पालकी ढोने वाले (कहार) का वेश धारण कर उसके स्वागत के लिए उपस्थित हुए। उस पंडित को लेशमात्र भी भनक नहीं थी कि पालकी उठाने वाला स्वयं महाकवि कालिदास है!

पालकी को अपने कंधे पर उठाकर कालिदास उस पंडित को लेकर चल पड़े। उस समय शिशिर (कड़ाके की ठंड) का मौसम था और बर्फीली हवाएँ शरीर को चीर रही थीं।
ठंड से ठिठुरते हुए उस पंडित ने संस्कृत में कहा: \"शीतं बहु बाधति\" (सर्दी बहुत कष्ट दे रही है / सता रही है)।

चतुर कालिदास ने तुरंत पलटकर प्रत्युत्तर दिया:
\"न तथा बाधते शीतं यथा बाधति बाधते\"
(अर्थात्: मुझे यह कड़ाके की सर्दी उतना कष्ट नहीं दे रही, जितना आपका अशुद्ध व्याकरण शब्द 'बाधति' कष्ट दे रहा है!)

उस पंडित को यह ज्ञात नहीं था कि संस्कृत में 'बाध्' धातु केवल 'आत्मनेपदी' होती है (जिसका शुद्ध रूप 'बाधते' होता है, 'बाधति' नहीं)।
पंडित मन ही मन भयभीत होकर सोचने लगा: \"यदि इस राज्य में पालकी उठाने वाले साधारण कहार भी पाणिनीय संस्कृत व्याकरण के इतने बड़े मर्मज्ञ हैं, तो राजदरबार के महापंडितों से शास्त्रार्थ करने पर तो मेरी भारी पराजय और घोर अपमान निश्चित है!\"

उसने तुरंत पालकी वाहक (कालिदास) को आदेश दिया: \"मुझे इस राज्य में आगे नहीं जाना है; पालकी मोड़ो, हम सीधे अपने घर वापस लौटेंगे!\""""
    }
}
