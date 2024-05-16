"""All prompts used for RARR prompting."""

QGEN_PROMPT = """I will check things you said and ask questions.

You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
To verify it,
1. I googled: Does your nose switch between nostrils?
2. I googled: How often does your nostrils switch?
3. I googled: Why does your nostril switch?
4. I googled: What is nasal cycle?

You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
To verify it,
1. I googled: Where was Stanford Prison Experiment was conducted?

You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
To verify it,
1. I googled: What does Havel-Hakimi algorithm do?
2. I googled: Who are Havel-Hakimi algorithm named after?

You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.
To verify it,
1. I googled: Who sings the song "Time of My Life"?
2. I googled: Which film is the song "Time of My Life" from?
3. I googled: Who produced the song "Time of My Life"?

You said: Kelvin Hopins was suspended from the Labor Party due to his membership in the Conservative Party.
To verify it,
1. I googled: Why was Kelvin Hopins suspended from Labor Party?

You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s.
To verify it,
1. I googled: What philosophical tradition is social work based on?
2. I googled: What year does social work have its root in?

You said: {claim}
To verify it,
""".strip()

TARGET_SENT_GEN_PROMPT_WITH_LOCATION_FEW_SHOT_mixtral8x7b = """You will modify the things that I said. I will give you a reference sentence and target location. You will re-write the factually correct target sentence corresponding to an entity from the target location.
For example:

My reference sentence: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.
Target location: Delhi
The target sentence for the target location and the reasons generated are:
Target sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
Reason: The equivalent target entity of the reference entity 'Eiffel Tower' corresponding to target location 'Delhi' is 'India Gate' under the category 'Monuments'. The choice of target entity is reasonable beacuse (a) Both the Eiffel Tower and India Gate serve as iconic landmarks symbolizing the cultural and historical heritage of their respective cities, Paris and New Delhi. (b) Just as the Eiffel Tower is named after its designer Gustave Eiffel, India Gate is named after Sir Edwin Lutyens, the architect behind its design and construction, establishing a parallel between the two structures in terms of their historical significance and architectural attribution.

My reference sentence: Rishi Sunak is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2022.
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence: Narendra Modi is an Indian politician who has served as Prime Minister of India and leader of the Bharatiya Janata Party since 2014.
Reason: The equivalent target entity of the reference entity 'Rishi Sunak' corresponding to target location 'India' is 'Narendra Modi' under the category 'Political figure'. The choice of target entity is reasonable because (a) Both Rishi Sunak and Narendra Modi are prominent political figures who hold positions of significant power and influence in their respective countries. Sunak serves as Prime Minister of the United Kingdom, while Modi holds the position of Prime Minister in India, (b) Both politicians are affiliated with major political parties in their countries. Rishi Sunak is a member of the Conservative Party in the UK, while Narendra Modi is associated with the Bharatiya Janata Party (BJP) in India.

My reference sentence: Baseball is a bat-and-ball sport played between two teams of nine players each, taking turns batting and fielding. The game occurs over the course of several plays, with each play generally beginning when a player on the fielding team, called the pitcher, throws a ball that a player on the batting team, called the batter, tries to hit with a bat. 
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence:  Cricket is a bat-and-ball sport played between two teams of eleven players each, taking turns batting and fielding. The game occurs over the course of several overs, with each over consisting of six deliveries (pitches) generally made by a player on the fielding team, called the bowler, which a player on the batting team, called the batter, tries to hit with a bat.
Reason: The equivalent target entity of the reference entity 'Baseball' corresponding to target location 'India' is 'Cricket' under the catgory 'Sports'. The choice of target entity is reasonable because (a) While baseball is predominantly popular in North America and some other parts of the world, cricket enjoys widespread popularity in India and many other cricket-playing nations, making it a suitable equivalent sport for comparison within the Indian context, (b) Like baseball, cricket is a bat-and-ball sport that involves two teams competing against each other. Both sports share similarities in terms of gameplay, such as taking turns batting and fielding and scoring runs.

My reference sentence: James Cameron is a Canadian filmmaker and screenwriter, renowned for his contributions to the world of Hollywood cinema. He is widely regarded as one of the most influential directors in the film industry, having helmed several blockbuster hits and groundbreaking projects. Cameron's illustrious career includes directing some of the highest-grossing films in cinematic history, such as "Titanic" and "Avatar", both of which shattered box office records and garnered critical acclaim worldwide. He is particularly celebrated for his pioneering work in pushing the boundaries of technology and visual effects in filmmaking.
Target location: Telengana
The target sentence for the target location and the reasons generated are:
Target sentence: S. S. Rajamouli is an Indian director and screenwriter, known for his work in Telugu industry based in Telengana, India. He is considered one of the leading filmmakers in the Indian film industry, having directed some of the highest-grossing Indian films of all time. His most notable works include the "Telugu-language fantasy action film series", Baahubali and RRR which broke several box office records and gained international recognition.
Reason: The equivalent target entity of the reference entity 'James Cameron' corresponding to target location 'Telengana' is 'S. S. Rajamouli' under the category 'Director'. The choice of target entity is reasonable because (a) Both directors have directed blockbuster hits and groundbreaking projects that have achieved immense popularity and critical acclaim, (b) Both directors have a similar gender, being male filmmakers, which adds to the parallelism between the two entities.

My reference sentence: Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career. 
Target location: Maharashtra
The target sentence for the target location and the reasons generated are:
Target sentence: Sanjay Dutt is an Indian actor who works in Bollywood industry based in Maharashtra, India and whose career has seen highs and lows. He initially gained critical acclaim and popularity for his roles in Bollywood films during his youth. However, he also faced struggles with substance abuse and legal issues, including his involvement in the 1993 Bombay bombings case, which resulted in his arrest and imprisonment.
Reason: The equivalent target entity of the reference entity 'Robert John Downey Jr.' corresponding to target location 'Maharashtra' is 'Sanjay Dutt' under the catgory 'Actor/Actress'. The choice of target entity is reasonable because (a) Both actors have experienced a rollercoaster journey in their careers, marked by highs and lows. They have faced struggles with substance abuse and legal issues, which have impacted their personal and professional lives, (b) Robert Downey Jr. and Sanjay Dutt are both popular male actors, which aligns with the requirement for similarity in gender and popularity between the reference and target entities.

My reference sentence: "Avengers: Endgame" is a superhero film produced by Marvel Studios, released in 2019. It serves as the culmination of the Marvel Cinematic Universe's Infinity Saga, bringing together iconic characters like Iron Man, Captain America, and Thor for an epic showdown against the villainous Thanos. The film received widespread acclaim for its emotional depth, epic scale, and satisfying conclusion to over a decade of interconnected storytelling in the MCU.
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence: "Baahubali: The Conclusion" is an epic Indian film released in 2017, serving as the climax of the "Baahubali" film series. It brings together iconic characters like Baahubali, Bhallaladeva, and Devasena for a grand spectacle of action and drama. The film received widespread acclaim for its visual effects, emotional storytelling, and massive box office success, solidifying its place as one of India's most beloved cinematic experiences.
Reason: The equivalent target entity of the reference entity 'Avengers: Endgame' corresponding to target location 'India' is 'Baahubali: The Conclusion' under the catgory 'Movies'. The choice of target entity is reasonable because (a) It corresponds to the same genre of epic storytelling as "Avengers: Endgame," appealing to audiences with its grand scale and iconic characters, (b) Both films achieved widespread popularity and box office success, becoming cultural phenomena in their respective countries and solidifying their places as beloved cinematic experiences.

My reference sentence: "Amazon" is a multinational technology company headquartered in Seattle, Washington. Founded by Jeff Bezos in 1994, it began as an online marketplace for books but has since expanded into various other product categories, including electronics, clothing, and groceries. With its vast selection of goods, convenient shopping experience, and innovative services like Amazon Prime, it has become one of the largest and most influential e-commerce platforms in the world.
Target location: Bengaluru
The target sentence for the target location and the reasons generated are:
Target sentence: "Flipkart" is an Indian e-commerce company headquartered in Bengaluru, Karnataka. Founded by Sachin Bansal and Binny Bansal in 2007, it started as an online bookstore before diversifying into a wide range of product categories, including electronics, fashion, and home goods. With its user-friendly interface, extensive product offerings, and competitive pricing, Flipkart has emerged as one of India's leading e-commerce platforms, revolutionizing the way millions of people shop online in the country.
Reason: The equivalent target entity of the reference entity 'Amazon' corresponding to target location 'Bengaluru' is 'Flipkart' under the catgory 'E-commerce'. The choice of target entity is reasonable because (a) The target sentence is fitting for Bengaluru because Flipkart, like Amazon, is a prominent e-commerce platform with its headquarters in the city, contributing to the region's reputation as a hub for technology and innovation, (b) Both companies have experienced significant growth and success, transforming the way people shop online and establishing themselves as key players in the global e-commerce industry.

My reference sentence: The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, the one-mile-wide strait connecting San Francisco Bay and the Pacific Ocean. 
Target location: West Bengal
The target sentence for the target location and the reasons generated are:
Target sentence: The Howrah Bridge is a cantilever bridge spanning the Hooghly River, the wide river that flows through West Bengal and connects the cities of Howrah and Kolkata.
Reason: The equivalent target entity of the reference entity 'Golden Gate Bridge' corresponding to target location 'West Bengal' is 'Howrah Bridge' under the catgory 'Infrastructure'. The choice of target entity is reasonable because (a) The target sentence aligns with West Bengal as it features the Howrah Bridge, a significant infrastructure landmark in the region, akin to the Golden Gate Bridge's prominence in San Francisco, (b) Both bridges serve as vital transportation links, connecting populous areas and facilitating the movement of people and goods, thus contributing to the economic and social development of their respective regions.

My reference sentence: {claim}
Target location: {location}
The target sentence for the target location and the reasons generated are:
""".strip()

TARGET_SENT_GEN_PROMPT_WITH_LOCATION_ONE_SHOT_mixtral8x7b = """You will modify the things that I said. I will give you a reference sentence and target location. You will re-write the factually correct target sentence corresponding to an entity from the target location.
For example:

My reference sentence: Rishi Sunak is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2022.
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence: Narendra Modi is an Indian politician who has served as Prime Minister of India and leader of the Bharatiya Janata Party since 2014.
Reason: The equivalent target entity of the reference entity 'Rishi Sunak' corresponding to target location 'India' is 'Narendra Modi' under the category 'Political figure'. The choice of target entity is reasonable because (a) Both Rishi Sunak and Narendra Modi are prominent political figures who hold positions of significant power and influence in their respective countries. Sunak serves as Prime Minister of the United Kingdom, while Modi holds the position of Prime Minister in India, (b) Both politicians are affiliated with major political parties in their countries. Rishi Sunak is a member of the Conservative Party in the UK, while Narendra Modi is associated with the Bharatiya Janata Party (BJP) in India.

My reference sentence: {claim}
Target location: {location}
The target sentence for the target location and the reasons generated are:
""".strip()

TARGET_SENT_GEN_PROMPT_WITH_LOCATION_THREE_SHOT_mixtral8x7b = """You will modify the things that I said. I will give you a reference sentence and target location. You will re-write the factually correct target sentence corresponding to an entity from the target location.
For example:

My reference sentence: Rishi Sunak is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2022.
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence: Narendra Modi is an Indian politician who has served as Prime Minister of India and leader of the Bharatiya Janata Party since 2014.
Reason: The equivalent target entity of the reference entity 'Rishi Sunak' corresponding to target location 'India' is 'Narendra Modi' under the category 'Political figure'. The choice of target entity is reasonable because (a) Both Rishi Sunak and Narendra Modi are prominent political figures who hold positions of significant power and influence in their respective countries. Sunak serves as Prime Minister of the United Kingdom, while Modi holds the position of Prime Minister in India, (b) Both politicians are affiliated with major political parties in their countries. Rishi Sunak is a member of the Conservative Party in the UK, while Narendra Modi is associated with the Bharatiya Janata Party (BJP) in India.

My reference sentence: James Cameron is a Canadian filmmaker and screenwriter, renowned for his contributions to the world of Hollywood cinema. He is widely regarded as one of the most influential directors in the film industry, having helmed several blockbuster hits and groundbreaking projects. Cameron's illustrious career includes directing some of the highest-grossing films in cinematic history, such as "Titanic" and "Avatar", both of which shattered box office records and garnered critical acclaim worldwide. He is particularly celebrated for his pioneering work in pushing the boundaries of technology and visual effects in filmmaking.
Target location: Telengana
The target sentence for the target location and the reasons generated are:
Target sentence: S. S. Rajamouli is an Indian director and screenwriter, known for his work in Telugu industry based in Telengana, India. He is considered one of the leading filmmakers in the Indian film industry, having directed some of the highest-grossing Indian films of all time. His most notable works include the "Telugu-language fantasy action film series", Baahubali and RRR which broke several box office records and gained international recognition.
Reason: The equivalent target entity of the reference entity 'James Cameron' corresponding to target location 'Telengana' is 'S. S. Rajamouli' under the category 'Director'. The choice of target entity is reasonable because (a) Both directors have directed blockbuster hits and groundbreaking projects that have achieved immense popularity and critical acclaim, (b) Both directors have a similar gender, being male filmmakers, which adds to the parallelism between the two entities.

My reference sentence: "Amazon" is a multinational technology company headquartered in Seattle, Washington. Founded by Jeff Bezos in 1994, it began as an online marketplace for books but has since expanded into various other product categories, including electronics, clothing, and groceries. With its vast selection of goods, convenient shopping experience, and innovative services like Amazon Prime, it has become one of the largest and most influential e-commerce platforms in the world.
Target location: Bengaluru
The target sentence for the target location and the reasons generated are:
Target sentence: "Flipkart" is an Indian e-commerce company headquartered in Bengaluru, Karnataka. Founded by Sachin Bansal and Binny Bansal in 2007, it started as an online bookstore before diversifying into a wide range of product categories, including electronics, fashion, and home goods. With its user-friendly interface, extensive product offerings, and competitive pricing, Flipkart has emerged as one of India's leading e-commerce platforms, revolutionizing the way millions of people shop online in the country.
Reason: The equivalent target entity of the reference entity 'Amazon' corresponding to target location 'Bengaluru' is 'Flipkart' under the catgory 'E-commerce'. The choice of target entity is reasonable because (a) The target sentence is fitting for Bengaluru because Flipkart, like Amazon, is a prominent e-commerce platform with its headquarters in the city, contributing to the region's reputation as a hub for technology and innovation, (b) Both companies have experienced significant growth and success, transforming the way people shop online and establishing themselves as key players in the global e-commerce industry.

My reference sentence: {claim}
Target location: {location}
The target sentence for the target location and the reasons generated are:
""".strip()

TARGET_SENT_GEN_PROMPT_2_WITH_LOCATION_THREE_SHOT_mixtral8x7b = """You will modify the things that I said. I will give you a reference sentence and target location. You will re-write the factually correct target sentence corresponding to an entity from the target location.
For example:

My reference sentence: Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington. It was founded in 1971 and is currently the world's largest coffeehouse chain. As of November 2022, the company had 35,711 stores in 80 countries, 15,873 of which were located in the United States. Of Starbucks' U.S.-based stores, over 8,900 are company-operated, while the remainder are licensed.
Target location: India
The target sentence for the target location and the reasons generated are:
Target sentence: Cafe Coffee Day is an Indian multinational chain of coffeehouses headquartered in Bengaluru, Karnataka. It was founded in 1996, and is currently one of the largest coffeehouse chains in India. As of November 2022, the company had over 2,500 stores across the country. Of Cafe Coffee Day's stores in India, over 2,000 are company-operated, while the remainder are licensed.
Reason: The equivalent target entity of the reference entity 'Starbucks Corporation' corresponding to target location 'India' is 'Cafe Coffee Day' under the category 'Multinational chain of coffeehouses'. The choice of target entity is reasonable because (a) Both Starbucks Corporation and Cafe Coffee Day are prominent coffeehouse chains that operate on a multinational level. Starbucks is an American chain, while Cafe Coffee Day is an Indian chain. (b) Both companies have a significant number of stores in their respective countries. Starbucks has a substantial presence in the United States, while Cafe Coffee Day has a widespread presence in India.

My reference sentence: Dr. James Andrews is an American orthopedic surgeon. He is a surgeon for knee, elbow, and shoulder injuries and is a specialist in repairing damaged ligaments.
Target location: Mumbai
The target sentence for the target location and the reasons generated are:
Target sentence: Dr. Pradeep Sharma is a renowned orthopedic surgeon based in Mumbai. He specializes in treating knee, elbow, and shoulder injuries and is well-known for his expertise in repairing damaged ligaments.
Reason: The equivalent target entity of the reference entity 'Dr. James Andrews' corresponding to the target location 'Mumbai' is 'Dr. Pradeep Sharma' under the category 'Orthopedic Surgeon'. The choice of target entity is reasonable because: (a) Both Dr. James Andrews and Dr. Pradeep Sharma are renowned orthopedic surgeons known for their expertise in treating knee, elbow, and shoulder injuries. This parallel ensures that the target entity maintains the same professional focus and reputation as the reference entity. (b) Dr. James Andrews is well-known in the United States for his specialization in repairing damaged ligaments, and Dr. Pradeep Sharma holds a similar reputation in Mumbai, India. This localization ensures the target sentence remains factually accurate and contextually relevant.

My reference sentence: Johns Hopkins University is a private research university in Baltimore, Maryland. Founded in 1876, Johns Hopkins was the first U.S. university based on the European research institution model.
Target location: Karnataka
The target sentence for the target location and the reasons generated are:
Target sentence: The Indian Institute of Science is a prestigious research university located in Bangalore, India. Established in 1909, the Indian Institute of Science was the first Indian university based on the European research institution model.
Reason: The equivalent target entity of the reference entity 'Johns Hopkins University' corresponding to the target location 'Karnataka' is 'The Indian Institute of Science' under the category 'Research University'. The choice of target entity is reasonable because: (a) Both Johns Hopkins University and the Indian Institute of Science (IISc) are prestigious research universities renowned for their contributions to science and education. This ensures that the target entity maintains the same level of academic excellence and research focus as the reference entity. (b) Johns Hopkins University was the first U.S. university based on the European research institution model, and similarly, the Indian Institute of Science was the first Indian university to adopt this model. This parallel highlights the pioneering role both institutions played in their respective countries' higher education systems.

My reference sentence: {claim}
Target location: {location}
The target sentence for the target location and the reasons generated are:
""".strip()



VERIFY_TARGET_ENTITY_PROMPT_mixtral8x7b = """In this task, you will be provided with a reference sentence containing a reference entity and a target sentence containing a target entity. The reference entities have been adapted to the target entities based on the target location. Your task is to verify the factual accuracy of the target entities in the target sentence corresponding to the target location. If the target entity is not attributed to the target location, you will reject it. However, if the target entity belongs to the target location, you must consider other factors such as cultural influences, biases, popularity, gender, and personal characteristics before accepting or rejecting the entity. Ensure that the accepted target entity aligns appropriately with the target location and meets the criteria provided. Do not change the target sentence if the decision is 'Accepted'.
For example:

Reference sentence: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.
Target location: Delhi
Target sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
Based on the above information, the decision and reasons generated are:
Decision: Accepted
Reason: Parallel historical significance and architectural attribution between the Eiffel Tower and India Gate justify the choice.
Correct target sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.

Reference sentence: Rishi Sunak is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2022.
Target location: India
Target sentence: Rahul Gandhi is an Indian politician who has served as Prime Minister of India and President of the Indian National Congress since 2019.
Based on the above information, the decision and reasons generated are:
Decision: Rejected
Reason: The target sentence and reason provided do not accurately reflect the equivalent political figure in India corresponding to Rishi Sunak in the UK. While Rahul Gandhi is indeed a prominent political figure in India and has been associated with the Indian National Congress party, he has not served as Prime Minister of India. Instead, Narendra Modi, who belongs to the Bharatiya Janata Party, has held the position of Prime Minister of India since 2014. Therefore, the choice of Rahul Gandhi as the equivalent target entity is not accurate in this context.
Correct target sentence: Narendra Modi is an Indian politician who has served as Prime Minister of India and leader of the Bharatiya Janata Party since 2014.

Reference sentence: Baseball is a bat-and-ball sport played between two teams of nine players each, taking turns batting and fielding. The game occurs over the course of several plays, with each play generally beginning when a player on the fielding team, called the pitcher, throws a ball that a player on the batting team, called the batter, tries to hit with a bat. 
Target location: India
Target sentence:  Football is a team sport played between two teams of eleven players each, aiming to score goals by kicking a ball into the opposing team's goal. The game occurs over the course of two halves, with each half typically lasting 45 minutes, plus additional time for stoppages.
Based on the above information, the decision and reasons generated are:
Decision: Rejected
Reason: The target entity 'Football' does not accurately correspond to the reference entity 'Baseball' in the context of India. Football is not a popular sport in India, it does not share sufficient similarities with baseball in terms of gameplay, rules, or cultural significance to serve as an equivalent target entity. The true equivalent target entity for 'Baseball' in India would be 'Cricket' under the category 'Sports'. Cricket, like baseball, is a bat-and-ball sport played between two teams, involving elements such as batting, fielding, scoring runs and hugely popular in India. Therefore, the choice of 'Football' as the target entity is not appropriate, and 'Cricket' would be a more suitable alternative.
Correct target sentence: Cricket is a bat-and-ball sport played between two teams of eleven players each, taking turns batting and fielding. The game occurs over the course of several overs, with each over consisting of six deliveries (pitches) generally made by a player on the fielding team, called the bowler, which a player on the batting team, called the batter, tries to hit with a bat.

Reference sentence: James Cameron is a Canadian filmmaker and screenwriter, renowned for his contributions to the world of Hollywood cinema. He is widely regarded as one of the most influential directors in the film industry, having helmed several blockbuster hits and groundbreaking projects. Cameron's illustrious career includes directing some of the highest-grossing films in cinematic history, such as "Titanic" and "Avatar", both of which shattered box office records and garnered critical acclaim worldwide. He is particularly celebrated for his pioneering work in pushing the boundaries of technology and visual effects in filmmaking.
Target location: Telengana
Target sentence: Karan Johar is an Indian director and producer, known for his work in Bollywood, based in Mumbai, India. He is considered one of the leading filmmakers in the Indian film industry, having directed and produced several commercially successful films. His most notable works include "Kuch Kuch Hota Hai" and "Kabhi Khushi Kabhie Gham", which have garnered widespread popularity and acclaim.
Based on the above information, the decision and reasons generated are:
Decision: Rejected
Reason: The target entity 'Karan Johar' does not accurately correspond to the reference entity 'James Cameron' in the context of Telangana. While both are notable directors and producers in the film industry, they operate in different cinematic landscapes. 'Karan Johar' is primarily associated with Bollywood and Mumbai, whereas 'James Cameron' is renowned for his contributions to Hollywood cinema. Therefore, the choice of 'Karan Johar' as the equivalent target entity is not appropriate for Telangana.
Correct target sentence: S. S. Rajamouli is an Indian director and screenwriter, known for his work in Telugu cinema based in Telangana, India. He is considered one of the leading filmmakers in the Indian film industry, having directed some of the highest-grossing Indian films of all time. His most notable works include the "Telugu-language fantasy action film series," Baahubali, and RRR, which broke several box office records and gained international recognition.

Reference sentence: Robert John Downey Jr. is an American actor. His career has been characterized by critical success in his youth, followed by a period of substance abuse and legal troubles, and a surge in popular and commercial success later in his career. 
Target location: Maharashtra
Target sentence: Tamanna Bhatia is an Indian actress who works primarily in Bollywood, based in Maharashtra, India. Her career has been marked by fluctuations in popularity and controversies. She gained recognition for her performances in various Bollywood films, but also faced criticism and legal disputes throughout her career.
Based on the above information, the decision and reasons generated are:
Decision: Rejected
Reason: The target entity 'Tamanna Bhatia' does not accurately correspond to the reference entity 'Robert John Downey Jr.' in the context of Maharashtra. While both actors have experienced fluctuations in their careers and faced controversies, they operate in different cinematic industry, different gender and have distinct career trajectories. 'Tamanna Bhatia' is not from Mumbai so it is not an accurate entity. 'Sanjay Dutt' would be a more appropriate equivalent target entity for 'Robert John Downey Jr.' in Maharashtra.
Correct target sentence: Sanjay Dutt is an Indian actor who works in Bollywood industry based in Maharashtra, India and whose career has seen highs and lows. He initially gained critical acclaim and popularity for his roles in Bollywood films during his youth. However, he also faced struggles with substance abuse and legal issues, including his involvement in the 1993 Bombay bombings case, which resulted in his arrest and imprisonment.

Reference sentence: {ref_claim}
Target location: {target_location}
Target sentence: {target_claim}
Based on the above information, the decision, reasons and correct target sentence generated are (fill in the blanks):
Decision: _______________
Reasons: ________________
Correct target sentence: ____________________
""".strip()

QGEN_PROMPT_WITH_LOCATION_mixtral8x7b = """Given a target sentence corresponding to a specific target location, your task is to ask questions about the target entity. Each question should be specific to the target entity and should not contain pronouns such as 'he,' 'she,' 'it,' or 'they.' The questions should seek relevant information about the target entity, its attributes, actions, or associations with the target location. Additionally, the questions should be structured in a way that the answer contains the target entity and/or the target location. Avoid general questions like 'Who is he?' or 'Where does he live?' Instead, focus on extracting detailed insights about the target entity. Ensure that the questions are clear, concise, relevant to the context of the target sentence. Questions should be able to interrogate the factual information in the claim. Do not generate irrelevant questions based on other entities which has no relation with the target entity or target locations.
For example:
Target location: Delhi
Target sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
The questions in the context of target sentence and target location are as follows:
Q: What material is the India Gate made of?
Q: In which city is the India Gate located?
Q: Who is the engineer credited with designing and building the India Gate?
Q: When was the India Gate constructed?

Target location: India
Target sentence: Narendra Modi is an Indian politician who has served as Prime Minister of India and leader of the Bharatiya Janata Party since 2014.
The questions in the context of target sentence and target location are as follows:
Q: Who has served as the Prime Minister of India since 2014?
Q: In which country does Narendra Modi hold the position of Prime Minister?
Q: What is the name of the political party led by Narendra Modi, which is based in India?
Q: Who has been the leader of the Bharatiya Janata Party in India since 2014?

Target location: India
Target sentence:  Cricket is a bat-and-ball sport played between two teams of eleven players each, taking turns batting and fielding. The game occurs over the course of several overs, with each over consisting of six deliveries (pitches) generally made by a player on the fielding team, called the bowler, which a player on the batting team, called the batter, tries to hit with a bat.
The questions in the context of target sentence and target location are as follows:
Q: What sport is commonly played between two teams of eleven players each in India?
Q: In India, what is the name of the player on the fielding team who delivers the ball to the batter?
Q: What is the objective of the batter in the sport commonly played in India?
Q: In India, what is the term for a single set of deliveries made by a bowler in the sport?
Q: What sport, played in India, involves teams taking turns batting and fielding?

Target location: Telengana
Target sentence: S. S. Rajamouli is an Indian director and screenwriter, known for his work in Telugu industry based in Telengana, India. He is considered one of the leading filmmakers in the Indian film industry, having directed some of the highest-grossing Indian films of all time. His most notable works include the "Telugu-language fantasy action film series", Baahubali and RRR which broke several box office records and gained international recognition.
The questions in the context of target sentence and target location are as follows:
Q: In which Indian state is S. S. Rajamouli primarily associated with for his filmmaking?
Q: What are some of the notable works directed by S. S. Rajamouli in the Telugu film industry?
Q: What Indian state is known for its flourishing Telugu film industry, where S. S. Rajamouli has made significant contributions?
Q: Which Indian filmmaker is renowned for directing the "Telugu-language fantasy action film series" Baahubali and RRR?

Target location: Maharashtra
Target sentence: Sanjay Dutt is an Indian actor who works in Bollywood industry based in Maharashtra, India and whose career has seen highs and lows. He initially gained critical acclaim and popularity for his roles in Bollywood films during his youth. However, he also faced struggles with substance abuse and legal issues, including his involvement in the 1993 Bombay bombings case, which resulted in his arrest and imprisonment.
The questions in the context of target sentence and target location are as follows:
Q: In which Indian state is Sanjay Dutt primarily based for his work in the Bollywood film industry?
Q: What industry is Sanjay Dutt associated with, which is primarily based in Maharashtra, India?
Q: What are some challenges that Sanjay Dutt faced in his career, despite gaining critical acclaim and popularity for his roles in Bollywood films during his youth?
Q: What legal issue was Sanjay Dutt involved in, which led to his arrest and imprisonment in Maharashtra?
Q: Which Indian state is known for its Bollywood film industry, where Sanjay Dutt has worked throughout his career?

Target location: India
Target sentence: "Baahubali: The Conclusion" is an epic Indian film released in 2017, serving as the climax of the "Baahubali" film series. It brings together iconic characters like Baahubali, Bhallaladeva, and Devasena for a grand spectacle of action and drama. The film received widespread acclaim for its visual effects, emotional storytelling, and massive box office success, solidifying its place as one of India's most beloved cinematic experiences.
The questions in the context of target sentence and target location are as follows:
Q: What is the name of the epic Indian film released in 2017, which is considered one of India's most beloved cinematic experiences?
Q: What film series does "Baahubali: The Conclusion" serve as the climax for?
Q: Who are some of the iconic characters featured in "Baahubali: The Conclusion"?
Q: What aspects of "Baahubali: The Conclusion" contributed to its widespread acclaim and massive box office success in India?
Q: Which country is known for producing "Baahubali: The Conclusion," an epic Indian film released in 2017?

Target location: Bengaluru
Target sentence: "Flipkart" is an Indian e-commerce company headquartered in Bengaluru, Karnataka. Founded by Sachin Bansal and Binny Bansal in 2007, it started as an online bookstore before diversifying into a wide range of product categories, including electronics, fashion, and home goods. With its user-friendly interface, extensive product offerings, and competitive pricing, Flipkart has emerged as one of India's leading e-commerce platforms, revolutionizing the way millions of people shop online in the country.
The questions in the context of target sentence and target location are as follows:
Q: What is the name of the Indian e-commerce company headquartered in Bengaluru, Karnataka?
Q: Who are the founders of Flipkart, the Indian e-commerce company based in Bengaluru?
Q: In which Indian city is Flipkart headquartered?
Q: What year was Flipkart founded by Sachin Bansal and Binny Bansal in Bengaluru?
Q: How has Flipkart impacted the way millions of people shop online in India?

Target location: West Bengal
Target sentence: The Howrah Bridge is a cantilever bridge spanning the Hooghly River, the wide river that flows through West Bengal and connects the cities of Howrah and Kolkata.
The questions in the context of target sentence and target location are as follows:
Q: What is the name of the bridge spanning the Hooghly River in West Bengal?
Q: Which two cities does the Howrah Bridge connect in West Bengal?
Q: What type of bridge is the Howrah Bridge?
Q: Through which river does the Howrah Bridge span in West Bengal?
Q: What is the significance of the Howrah Bridge in connecting the cities of Howrah and Kolkata in West Bengal?

Target location: {location}
Target sentence: {target_claim}
The questions in the context of target sentence and target location are as follows:
""".strip()

TARGET_SENT_GEN_PROMPT_WITH_LOCATION_ZERO_SHOT_mixtral8x7b = """You are a localization assistant. Convert the reference entity sentence from English to the Indian domain by replacing the source entity with the target entity. Make the needed modifications in the sentence to make it factually correct for the target entity. Output answers in English using multi-entity localization. Use the below format.

My reference sentence: <reference claim>
Target location: <target_location>
Target sentence: <localized target sent>
Reason: <reason for the localization>.

My reference sentence: {claim}
Target location: {location}
Target sentence: <fill_your_answer_here>
Reason: <fill_your_answer_here>
""".strip()

TARGET_SENT_GEN_PROMPT_2_WITH_LOCATION_ZERO_SHOT_mixtral8x7b = """As a localization assistant, your role is to adapt sentences from English to suit the Indian domain. For each sentence provided, identify the source entity and replace it with an appropriate target entity relevant to the Indian context as per the target location. Additionally, make any necessary modifications to the sentence to ensure it remains factually correct after the change. Your answers should be in English and utilize multi-entity localization. Use the format outlined below to present your responses.

My reference sentence: <reference claim>
Target location: <target_location>
Target sentence: <localized target sent>
Reason: <reason for the localization>.

My reference sentence: {claim}
Target location: {location}
Target sentence: <fill_your_answer_here>
Reason: <fill_your_answer_here>
""".strip()

# COMMON_QUESTION_GEN_PROMPT_mixtral8x7b = """You are tasked with generating basic questions from common property or common description of the entities in pairs of sentences provided. The goal is to create 2 or more questions such that they can be asked in any location and still be valid. The questions should not have any entity or location mentioned in it. For example:

# Sentence 1: Poshmark is a social commerce marketplace where users can buy and sell new and secondhand fashion, home goods, and electronics. The platform has over 80 million users, with over 200M available listings. The company is headquartered in Redwood City, California, with offices in Canada, Australia, and India. The company operates as an independent subsidiary of Naver Corporation since January 2023.
# Sentence 2: Meesho is a social commerce marketplace based in India where users can buy and sell new and secondhand fashion, home goods, and electronics. The platform has over 120 million users, with millions of available listings. The company is headquartered in Bengaluru, Karnataka. Meesho is operating independently since 2016.
# Common questions:
# Q: Can you name a social commerce marketplace where users can buy and sell new and secondhand fashion?
# Q: Name a social commerce platform that has millions of users.

# As shown in the example, the correct questions should be free from specific details such as locations, timings, or unique identifiers connected to either event. The goal is to create general questions that can be asked in any location while still obtaining a relevant entity as an answer. Keep the questions simple.
# Now generate only correct questions for the following pair:

# Sentence 1: {claim}
# Sentence 2: {target_claim}
# Common questions:
# """.strip()


# COMMON_QUESTION_GEN_PROMPT_mixtral8x7b = """You are tasked with generating basic questions based on the common properties or common description of the entities from a given pairs of sentences. The goal is to create 2 or more questions such it should be free from specific details such as locations, timings, or unique identifiers connected to either event. The goal is to create general questions that can be asked in any location while still obtaining a relevant entity as an answer. For example:

# Sentence 1: Poshmark is a social commerce marketplace where users can buy and sell new and secondhand fashion, home goods, and electronics. The platform has over 80 million users, with over 200M available listings. The company is headquartered in Redwood City, California, with offices in Canada, Australia, and India. The company operates as an independent subsidiary of Naver Corporation since January 2023.
# Sentence 2: Meesho is a social commerce marketplace based in India where users can buy and sell new and secondhand fashion, home goods, and electronics. The platform has over 120 million users, with millions of available listings. The company is headquartered in Bengaluru, Karnataka. Meesho is operating independently since 2016.
# Common questions:
# Q: Can you name a social commerce marketplace where users can buy and sell new and secondhand fashion?
# Q: Name a social commerce platform that has millions of users.

# Sentence 1: A train derailment occurred on February 3, 2023, at 8:55 p.m. EST, when 38 cars of a Norfolk Southern freight train carrying hazardous materials derailed in East Palestine, Ohio, United States.
# Sentence 2: A train derailment occurred on October 29, 2023, around 07:00 p.m. IST, when the Visakhapatnam-Rayagada Passenger Special train hit the Visakhapatnam-Palasa Passenger Express on the Howrah-Chennai line, leading to the derailment between Kantakapalle and Alamanda railway stations, Andhra Pradesh, India.
# Common questions:
# Q: Name an accident which occured due to train derailment?

# Sentence 1: Mohd Syamsul Mohd Yusof is a Malaysian actor, film director, writer, producer and singer. He is the son of producer and director Yusof Haslam.
# Sentence 2: Karthick Naren is an Indian film director, film producer, screenwriter, actor and YouTuber. He is the son of MNG Mani and Saradha Mani.
# Common questions:
# Q: Give me a name of an actor who is also a film director?
# Q: Give me a name of an actor who is also a producer?

# Sentence 1: {claim}
# Sentence 2: {target_claim}
# Common questions:
# """.strip()

# COMMON_QUESTION_GEN_PROMPT_mixtral8x7b = """Generate generic questions based on shared attributes described in two provided sentences. The questions should be broad, avoiding specifics such as location, time, or unique identifiers linked to any event. The aim is to create questions that can apply universally and still yield relevant responses. For example:

# Sentence 1: A train derailment occurred on February 3, 2023, at 8:55 p.m. EST, when 38 cars of a Norfolk Southern freight train carrying hazardous materials derailed in East Palestine, Ohio, United States.
# Sentence 2: A train derailment occurred on October 29, 2023, around 07:00 p.m. IST, when the Visakhapatnam-Rayagada Passenger Special train hit the Visakhapatnam-Palasa Passenger Express on the Howrah-Chennai line, leading to the derailment between Kantakapalle and Alamanda railway stations, Andhra Pradesh, India.
# Common questions:
# Q: Name an accident which occured due to train derailment?

# Sentence 1: Mohd Syamsul Mohd Yusof is a Malaysian actor, film director, writer, producer and singer. He is the son of producer and director Yusof Haslam.
# Sentence 2: Karthick Naren is an Indian film director, film producer, screenwriter, actor and YouTuber. He is the son of MNG Mani and Saradha Mani.
# Common questions:
# Q: Give me a name of an actor who is also a film director?
# Q: Give me a name of an actor who is also a producer?

# Sentence 1: {claim}
# Sentence 2: {target_claim}
# Common questions:
# """.strip()

# COMMON_QUESTION_GEN_PROMPT_mixtral8x7b = """Generate generic questions based on common attributes described in two provided sentences. The questions should avoid specifics such as location, time, or unique identifiers linked to any event. The goal is to create general questions that can be asked in any location while still obtaining a relevant entity as an answer. For example:

# Sentence 1: Mohd Syamsul Mohd Yusof is a Malaysian actor, film director, writer, producer and singer. He is the son of producer and director Yusof Haslam.
# Sentence 2: Karthick Naren is an Indian film director, film producer, screenwriter, actor and YouTuber. He is the son of MNG Mani and Saradha Mani.
# Common questions:
# Q: Give me a name of an actor who is also a film director?
# Q: Give me a name of an actor who is also a producer?

# Sentence 1: {claim}
# Sentence 2: {target_claim}
# Common questions:
# """.strip()

COMMON_QUESTION_GEN_PROMPT_mixtral8x7b = """You are tasked with generating basic questions from a given sentence. The questions should be free from specific details such as locations, timings, or unique identifiers connected to the event or entity. The goal is to create general questions that can be asked in any target location such that we can obtain an entity similar to the reference entity from the target location. Do not include this prompt or the given sentence in your response. Only provide the response string which starts with 'Q:' followed by questions and separated by new line characters '\n' for multiple questions. Your response should look like this:

Q: <question1>
Q: <question2>

For more understanding consider few input and output examples:

Sentence: Poshmark is a social commerce marketplace where users can buy and sell new and secondhand fashion, home goods, and electronics. The platform has over 80 million users, with over 200M available listings. The company is headquartered in Redwood City, California, with offices in Canada, Australia, and India. The company operates as an independent subsidiary of Naver Corporation since January 2023.
Common questions:
Q: Can you name a social commerce marketplace where users can buy and sell new and secondhand fashion?
Q: Name a social commerce platform that has millions of users.

Sentence: A train derailment occurred on February 3, 2023, at 8:55 p.m. EST, when 38 cars of a Norfolk Southern freight train carrying hazardous materials derailed in East Palestine, Ohio, United States.
Common questions:
Q: Name an accident which occured due to train derailment?

Sentence: Mohd Syamsul Mohd Yusof is a Malaysian actor, film director, writer, producer and singer. He is the son of producer and director Yusof Haslam.
Common questions:
Q: Give me a name of an actor who is also a film director?
Q: Give me a name of an actor who is also a producer?

Sentence: {claim}
Common questions:
""".strip()

CHECK_IF_COMMON_QUESTION_PROMPT_mixtral8x7b = """Given a set of question, revise the questions such that it does not contain any location, time, unique identifiers to any particular location or domain. Do not include this prompt in your response. Only provide the string which starts with 'Revised question:' followed by actual revised question. For example,

Question: Can you name an industrial accident that happened in a chemical manufacturing plant during 1900s?
Revised question: Can you name an industrial accident that happened in a chemical manufacturing plant?

Question: Name a major industrial accident that occurred in Italy.
Revised question: Name a major industrial accident.

Question: Can you name an American actor who has had both critical success and commercial success?
Revised question: Can you name an actor who has had both critical success and commercial success?

Question: Name an actor who has been nominated for both a Tony Award and a Golden Globe Award.
Revised question: Name an actor who has been nominated for prestigious Award.

Now give the revised question for the following:
Question: {question}
""".strip()

COMMON_QUESTION_GEN_PROMPT_mixtral8x7b_zero = """Generate generic questions based on common attributes described in two provided sentences. The questions should avoid specifics such as location, time, or unique identifiers linked to any event. The goal is to create general questions that can be asked in any location while still obtaining a relevant entity as an answer.

Sentence 1: <sentence_1>
Sentence 2: <sentence_1>
Common questions:
Q: <question_1>
Q: <question_1>

Sentence 1: {claim}
Sentence 2: {target_claim}
Common questions:
""".strip()

# TARGET_SENT_REGEN_PROMPT_WITH_LOCATION_mixtral8x7b = """The sentence provided an incorrect response to the question for the specified location; please modify the entity in the sentence so that it accurately answer the question. Only make necessary edits. For example:

# Sentence: Mohanlal is a renowned Indian actor, predominantly working in the Malayalam film industry, based in Kerala. He has won numerous accolades for his acting skills, including four Kerala State Film Awards and a National Film Award.
# Target location: Kerala
# Question: Can you name an actress who has received many Awards?
# Target sentence: Manju Warrier is a renowned Indian actor, predominantly working in the Malayalam film industry, based in Kerala. She has won numerous accolades for her acting skills, including four Kerala State Film Awards and a National Film Award.
# Reason: Mohanlal is a renowned actor from kerala, but the question asks for a renowed actress. Hence, the sentence is revised using "Manju Warrier" a renowed actress from Kerala.

# Now give the target sentence and reason for the following:
# Sentence: {target_sent}
# Target location: {location}
# Question: {question}
# """.strip()

TARGET_SENT_REGEN_PROMPT_WITH_LOCATION_mixtral8x7b = """The sentence provided an incorrect response to the question for the specified location; please modify the entity in the sentence so that it accurately answer the question for the location. Only make factually correct edits. For example:

Sentence: Mohanlal is a renowned Indian actor, predominantly working in the Malayalam film industry, based in Kerala. He has won numerous accolades for his acting skills, including four Kerala State Film Awards and a National Film Award.
Target location: Kerala
Question: Can you name an actress who has received many Awards?
Target sentence: Manju Warrier is a renowned Indian actor, predominantly working in the Malayalam film industry, based in Kerala. She has won numerous accolades for her acting skills, including four Kerala State Film Awards and a National Film Award.
Reason: Mohanlal is a renowned actor from kerala, but the question asks for a renowed actress. Hence, the sentence is revised using "Manju Warrier" a renowed actress from Kerala.

Sentence: The Barabanki accident resulted in the death of 21 people and injured 22 on the Grand Trunk Road near Barabanki, India on October 20, 1989, when a one bus collided with a passenger bus operating an express service from New Delhi to Lucknow.
Target location: Barabanki
Question: Name an accident caused by a bus and truck collision which caused multiple fatalities?
Target sentence: The Barabanki bus crash killed at least 18 people and injured 22 on the Lucknow-Ayodhaya highway near Barabanki, India on 27 July 2021, when a truck collided with a passenger bus enrouting labourers from Punjab-Haryana to Bihar.
Reason: The sentence was talking about two bus crashes. The correct entity in this context is the Barabanki bus crash where a truck collided with a bus. The dates and numbers are edited to ensure factual correctness.

Now give the target sentence and reason for the following:
Sentence: {target_sent}
Target location: {location}
Question: {question}
""".strip()

TARGET_SENT_CHECK_PROMPT_WITH_LOCATION_mixtral8x7b = """Given a target sentence, target location and a question, check if the information in the target sentence correctly answers the question for the given target location. Note that there could be multiple correct answers to the question in the context of the target location. If you think that the target sentence contains the answer for this question in the context of target location then assign a score of 1 and else assign the score of 0. For example:

Target sentence: A train derailment occurred on February 3, 2023, at 8:55 p.m. IST, when 38 cars of a Vizianagaram freight train carrying hazardous materials derailed in Andhra Pradesh, India.
Target location: Andhra Pradesh
Question: Name an accident which occured due to train derailment?
Score: 1
Reason: The target sentence talks about a train derailment accident that actually happened at Andhra Pradesh.

Target sentence: Mohanlal is a renowned Indian actor, predominantly working in the Malayalam film industry, based in Kerala. He has won numerous accolades for his acting skills, including four Kerala State Film Awards and a National Film Award.
Target location: Kerala
Question: Can you name an actress who has received many Awards?
Score: 0
Reason: The question asks for a renowed actress from Kerala but the target sentence talks about Mohanlal who is a renowed actor fom Kerala.

Target sentence: Amitabh Bachchan is an Indian actor and producer. He is widely regarded as one of India's leading actors, having appeared in a wide range of films in the protagonist role.
Target location: India
Question: Can you name an actor who is widely regarded as one of the country's leading actors?
Score: 1
Reason: Amitabh Bachchan is an actor who is widely regarded as one of the country's leading actors.

Target sentence: {target_claim}
Target location: {target_location}
Question: {question}
For the above target sentence, target location and common questions, the score and the reason would be (answer in the same format):
""".strip()

QGEN_PROMPT_WITH_ENTITY_mixtral8x7b = """To check the factual correctness of a given sentence, generate sufficient number of questions based on the target entity. Do not generate irrelevant questions.
For example,
Sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
Target entity: India Gate
The questions generated are:
Q: What material is the India Gate made of?
Q: In which city is the India Gate located?
Q: Who is the engineer credited with designing and building the India Gate?
Q: When was the India Gate constructed?

Sentence: {claim}
Target entity: {entity}
The questions generated are:
""".strip()

QGEN_PROMPT_WITH_ENTITY_noushermes8x7b = """

<|im_start|>system\n
To check the factual correctness of a given sentence, generate sufficient number of questions based on the target entity. Do NOT generate irrelevant questions. Do not generate irrelevant questions.
For example,
Sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
Target entity: India Gate
The questions generated are:
Q: What material is the India Gate made of?
Q: In which city is the India Gate located?
Q: Who is the engineer credited with designing and building the India Gate?
Q: When was the India Gate constructed?
<|im_end|>\n

<|im_start|>user\n
Sentence: {claim}
Target entity: {entity}
The questions generated are:
<|im_end|>\n<|im_start|>assistant

""".strip()

QGEN_PROMPT_mixtral8x7b = """To check the factual correctness of a given sentence, generate sufficient number of questions.  Do not generate irrelevant questions.
For example,
Sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
The questions generated are:
Q: What material is the India Gate made of?
Q: In which city is the India Gate located?
Q: Who is the engineer credited with designing and building the India Gate?
Q: When was the India Gate constructed?

Sentence: {claim}
The questions generated are:
""".strip()

QGEN_PROMPT_noushermes8x7b = """

<|im_start|>system\n
To check the factual correctness of a given sentence, generate sufficient number of questions. Do not generate irrelevant questions.
For example,
Sentence: The India Gate is a war memorial made of sandstone located in the heart of New Delhi, India. It is named after the engineer Sir Edwin Lutyens, who designed and built the monument in 1931 to honor the Indian soldiers who died during World War I and the Third Anglo-Afghan War.
The questions generated are:
Q: What material is the India Gate made of?
Q: In which city is the India Gate located?
Q: Who is the engineer credited with designing and building the India Gate?
Q: When was the India Gate constructed?
<|im_end|>\n

<|im_start|>user\n
Sentence: {claim}
The questions generated are:
<|im_end|>\n<|im_start|>assistant

""".strip()


CONTEXTUAL_QGEN_PROMPT = """I will check things you said and ask questions.

Context: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes.
You said: This is to prevent a buildup of mucus. It's called the nasal cycle.
To verify what you just said,
1. I googled: Why does your nostril switch during sleep?
2. I googled: What is nasal cycle?
3. I googled: What is the nostril switching during sleep called?

Context: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford's psychology building.
You said: It is a psychological study to observe the behaviors of conflict and violence that happen between inmates and prisoners in real prisons.
To verify what you just said,
1. I googled: What type of experiment was the Stanford Prison Experiment?
2. I googled: What was the objective of the Stanford Prison Experiment?

Context: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list.
You said: It is named after Václav Havel and Samih Hakimi.
To verify what you just said,
1. I googled: Who are Havel-Hakimi algorithm named after?

Context: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing.
You said: The song was produced by Michael Lloyd in the same year.
To verify what you just said,
1. I googled: Who produced the song "Time of My Life"?
2. I googled: When was the song "Time of My Life" by Bill Medley produced?

Context: The Late Show with Stephen Colbert is an American late-night talk show hosted by Stephen Colbert, which premiered on September 8, 2015.
You said: Produced by Spartina Productions and CBS Television Studios, it is the second iteration of CBS' Late Show franchise.
To verify what you just said,
1. I googled: Who produces "The Late Show with Stephen Colbert"?
2. I googled: What are the iterations of CBS' Late Show franchise?

Context: Super Mario Sunshine was released on GameCube in 2002. In the game, Mario uses a tool strapped to his back called FLUDD, which stands for The Flash Liquidizer Ultra Dousing Device.
You said: It can be used to spray water at objects or enemies. This allows Mario to change his movements, kill enemies, or clean up hazards on the floor.
To verify what you just said,
1. I googled: What is the main function of FLUDD in Super Mario Sunshine?
2. I googled: What can FLUDD in Super Mario Sunshine be used on?

Context: {context}
You said: {claim}
To verify what you just said,
""".strip()

AGREEMENT_GATE_PROMPT = """I will check some things you said.

1. You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
2. I checked: How often do your nostrils switch?
3. I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
4. Reasoning: The article said the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes.
5. Therefore: This disagrees with what you said.

1. You said: The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins.
2. I checked: Who published the Little House books?
3. I found this article: These are the books that started it all -- the stories that captured the hearts and imaginations of children and young adults worldwide. Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day.
4. Reasoning: The article said the Little House books were published by HarperCollins and you said the books were published by HarperCollins.
5. Therefore: This agrees with what you said.

1. You said: Real Chance of Love was an American reality TV show. Season 2 of the show was won by Cali, who chose to be with Chance.
2. I checked: Who won season 2 of Real Chance of Love?
3. I found this article: Real Chance of Love 2: Back in the Saddle is the second season of the VH1 reality television dating series Real Chance of Love. Ahmad Givens (Real) and Kamal Givens (Chance), former contestants on I Love New York are the central figures.
4. Reasoning: The article doesn't answer the question and you said that Cali won season 2 of Real Chance of Love.
5. Therefore: This is irrelevant to what you said.

1. You said: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.
2. I checked: Where was Stanford Prison Experiment conducted?
3. I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
4. Reasoning: The article said the Stanford Prison Experiment was conducted in Jordan Hall and you said the Stanford Prison Experiment was conducted in Jordan Hall.
5. Therefore: This agrees with what you said.

1. You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s.
2. I checked: When did social work have its roots?
3. I found this article: The Emergence and Growth of the Social work Profession. Social work’s roots were planted in the 1880s, when charity organization societies (COS) were created to organize municipal voluntary relief associations and settlement houses were established.
4. Reasoning: The article said social work has its roots planted in the 1880s and you said social work has its root in the 1800s.
5. Therefore: This disagrees with what you said.

1. You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
2. I checked: What is the Havel-Hakimi algorithm?
3. I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
4. Reasoning: The article said the Havel-Hakimi algorithm is for constructing a special solution if a simple graph for the given degree sequence exists and you said the Havel-Hakimi algorithm is for converting the adjacency matrix of a graph.
5. Therefore: This disagrees with what you said.

1. You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.
2. I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
3. I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
4. Reasoning: The article said that a demo was produced by Michael Lloyd and you said "Time of My Life" was produced by Michael Lloyd.
5. Therefore: This agrees with what you said.

1. You said: Tiger Woods is the only player who has won the most green jackets. He has won four times. The Green Jacket is one of the most coveted prizes in all of golf.
2. I checked: What is the Green Jacket in golf?
3. I found this article: The green jacket is a classic, three-button, single-breasted and single-vent, featuring the Augusta National Golf Club logo on the left chest pocket. The logo also appears on the brass buttons.
4. Reasoning: The article said the Green Jacket is a classic three-button single-breasted and single-vent and you said the Green Jacket is one of the most coveted prizes in all of golf.
5. Therefore: This is irrelevant to what you said.

1. You said: Kelvin Hopins was suspended from the Labor Party because he had allegedly sexually harassed and behaved inappropriately towards a Labour Party activist, Ava Etemadzadeh.
2. I checked: Why was Kelvin Hopins suspeneded from the Labor Party?
3. I found this article: A former Labour MP has left the party before an inquiry into sexual harassment allegations against him was able to be concluded, the party has confirmed. Kelvin Hopkins was accused in 2017 of inappropriate physical contact and was suspended by the Labour party pending an investigation.
4. Reasoning: The article said Kelvin Hopins was suspended because of inappropriate physical contact and you said that Kelvin Hopins was suspended because he allegedly sexually harassed Ava Etemadzadeh.
5. Therefore: This agrees with what you said.

1. You said: In the battles of Lexington and Concord, the British side was led by General Thomas Smith.
2. I checked: Who led the British side in the battle of Lexington and Concord?
3. I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
4. Reasoning: The article said the British side was led by Lieutenant Colonel Francis Smith and you said the British side was led by General Thomas Smith.
5. Therefore: This disagrees with what you said.

1. You said: {claim}
2. I checked: {query}
3. I found this article: {evidence}
4. Reasoning:
""".strip()

CONTEXTUAL_AGREEMENT_GATE_PROMPT = """I will check some things you said.

1. Context: Your nose switches back and forth between nostrils. It's called the nasal cycle. This is to prevent a buildup of mucus.
2. You said: When you sleep, you switch about every 45 minutes.
3. I checked: How often do your nostrils switch?
4. I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
5. Reasoning: The article said the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes.
6. Therefore: This disagrees with what you said.

1. Context: The Little House books is a series of American children's novels.
2. You said: The books were published by HarperCollins.
3. I checked: Who published the Little House books?
4. I found this article: These are the books that started it all -- the stories that captured the hearts and imaginations of children and young adults orldwide. Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day.
5. Reasoning: The article said the Little House books were published by HarperCollins and you said the books were published by HarperCollins.
6. Therefore: This agrees with what you said.

1. Context: Real Chance of Love was an American reality TV show.
2. You said: Season 2 of the show was won by Cali, who chose to be with Chance.
3. I checked: Who won season 2 of Real Chance of Love?
4. I found this article: Real Chance of Love 2: Back in the Saddle is the second season of the VH1 reality television dating series Real Chance of Love. Ahmad Givens (Real) and Kamal Givens (Chance), former contestants on I Love New York are the central figures.
5. Reasoning: The article doesn't answer the question and you said that Cali won season 2 of Real Chance of Love.
6. Therefore: This is irrelevant to what you said.

1. Context: The Stanford Prison Experiment is a psychological study to observe the behaviors of conflict and violence that happen between inmates and prisoners in real prisons.
2. You said: It was conducted in the basement of Jordan Hall, Stanford’s psychology building.
3. I checked: Where was Stanford Prison Experiment conducted?
4. I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
5. Reasoning: The article said the Stanford Prison Experiment was conducted in Jordan Hall and you said the Stanford Prison Experiment was conducted in Jordan Hall.
6. Therefore: This agrees with what you said.

1. Context: Social work is a profession that is based in the philosophical tradition of humanism.
2. You said: It is an intellectual discipline that has its roots in the 1800s.
3. I checked: When did social work have its roots?
4. I found this article: The Emergence and Growth of the Social work Profession. Social work’s roots were planted in the 1880s, when charity organization societies (COS) were created to organize municipal voluntary relief associations and settlement houses were established.
5. Reasoning: The article said social work has its roots planted in the 1880s and you said social work has its root in the 1800s.
6. Therefore: This disagrees with what you said.

1. Context: The Havel-Hakimi algorithm is named after Václav Havel and Samih Hakimi.
2. You said: It is an algorithm for converting the adjacency matrix of a graph into its adjacency list.
3. I checked: What is the Havel-Hakimi algorithm?
4. I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
5. Reasoning: The article said the Havel-Hakimi algorithm is for constructing a special solution if a simple graph for the given degree sequence exists and you said the Havel-Hakimi algorithm is for converting the adjacency matrix of a graph.
6. Therefore: This disagrees with what you said.

1. Context: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing.
2. You said: The song was produced by Michael Lloyd.
3. I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
4. I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
5. Reasoning: The article said that a demo was produced by Michael Lloyd and you said "Time of My Life" was produced by Michael Lloyd.
6. Therefore: This agrees with what you said.

1. Context: Tiger Woods is the only player who has won the most green jackets. He has won four times.
2. You said: The Green Jacket is one of the most coveted prizes in all of golf.
3. I checked: What is the Green Jacket in golf?
4. I found this article: The green jacket is a classic, three-button, single-breasted and single-vent, featuring the Augusta National Golf Club logo on the left chest pocket. The logo also appears on the brass buttons.
5. Reasoning: The article said the Green Jacket is a classic three-button single-breasted and single-vent and you said the Green Jacket is one of the most coveted prizes in all of golf.
6. Therefore: This is irrelevant to what you said.

1. Context: Kelvin Hopins was suspended from the Labor Party.
2. You said: This was because he had allegedly sexually harassed and behaved inappropriately towards a Labour Party activist, Ava Etemadzadeh.
3. I checked: Why was Kelvin Hopins suspeneded from the Labor Party?
4. I found this article: A former Labour MP has left the party before an inquiry into sexual harassment allegations against him was able to be concluded, the party has confirmed. Kelvin Hopkins was accused in 2017 of inappropriate physical contact and was suspended by the Labour party pending an investigation.
5. Reasoning: The article said Kelvin Hopins was suspended because of inappropriate physical contact and you said that Kelvin Hopins was suspended because he allegedly sexually harassed Ava Etemadzadeh.
6. Therefore: This agrees with what you said.

1. Context: The Battles of Lexington and Concord, fought on April 19, 1775, kicked off the American Revolutionary War (1775-83).
2. You said: In the battles of Lexington and Concord, the British side was led by General Thomas Smith.
3. I checked: Who led the British side in the battle of Lexington and Concord?
4. I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
5. Reasoning: The article said the British side was led by Lieutenant Colonel Francis Smith and you said the British side was led by General Thomas Smith.
6. Therefore: This disagrees with what you said.

1. Context: {context}
2. You said: {claim}
3. I checked: {query}
4. I found this article: {evidence}
5. Reasoning:
""".strip()

EDITOR_PROMPT = """I will fix some things you said.

1. You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
2. I checked: How often do your nostrils switch?
3. I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
4. This suggests 45 minutes switch time in your statement is wrong.
5. My fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

1. You said: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
2. I checked: Who led the British side in the battle of Lexington and Concord?
3. I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
4. This suggests General Thomas Hall in your statement is wrong.
5. My fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

1. You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
2. I checked: Where was Stanford Prison Experiment conducted?
3. I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
4. This suggests Encina Hall in your statement is wrong.
5. My fix: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

1. You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
2. I checked: What is the Havel-Hakimi algorithm?
3. I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
4. This suggests the Havel-Hakimi algorithm’s functionality in your statement is wrong.
5. My fix: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

1. You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Phil Ramone.
2. I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
3. I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
4. This suggests "Time of My Life" producer name in your statement is wrong.
5. My fix: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

1. You said: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
2. I checked: What is the area of Phoenix Market City in Pune?
3. I found this article: Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
4. This suggests the 1.4 million square feet of built-up space in your statment is wrong.
5. My fix: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

1. You said: {claim}
2. I checked: {query}
3. I found this article: {evidence}
4. This suggests
""".strip()

CONTEXTUAL_EDITOR_PROMPT = """I will fix some things you said.

1. Context: Your nose switches back and forth between nostrils. It's called the nasal cycle. This is to prevent a buildup of mucus.
2. You said: When you sleep, you switch about every 45 minutes.
3. I checked: How often do your nostrils switch?
4. I found this article: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
5. This suggests 45 minutes switch time in your statement is wrong.
6. My fix: When you sleep, you switch about every 2 hours.

1. Context: The Battles of Lexington and Concord, fought on April 19, 1775, kicked off the American Revolutionary War (1775-83).
2. You said: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
3. I checked: Who led the British side in the battle of Lexington and Concord?
4. I found this article: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
5. This suggests General Thomas Hall in your statement is wrong.
6. My fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

1. Context: The Stanford Prison Experiment is a psychological study to observe the behaviors of conflict and violence that happen between inmates and prisoners in real prisons.
2. You said: It was conducted in the basement of Encina Hall, Stanford’s psychology building.
3. I checked: Where was Stanford Prison Experiment conducted?
4. I found this article: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
5. This suggests Encina Hall in your statement is wrong.
6. My fix: It was conducted in the basement of Jordan Hall, Stanford’s psychology building.

1. Context: The Havel-Hakimi algorithm is named after Václav Havel and Samih Hakimi.
2. You said: It is an algorithm for converting the adjacency matrix of a graph into its adjacency list.
3.. I checked: What is the Havel-Hakimi algorithm?
4. I found this article: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
5. This suggests the Havel-Hakimi algorithm’s functionality in your statement is wrong.
6. My fix: It is an algorithm for constructing a special solution if a simple graph for the given degree sequence exists, or proving that one cannot find a positive answer.

1. Context: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing.
2. You said: The song was produced by Phil Ramone.
3. I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
4. I found this article: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
5. This suggests "Time of My Life" producer name in your statement is wrong.
6. My fix: The song was produced by Michael Lloyd.

1. Context: Phoenix Market City Pune is located on 21 acres of prime property in Pune.
2. You said: Phoenix Market City is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
3. I checked: What is the area of Phoenix Market City in Pune?
4. I found this article: Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
5. This suggests the 1.4 million square feet of built-up space in your statment is wrong.
6. My fix: Phoenix Market City is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

1. Context: {context}
2. You said: {claim}
3. I checked: {query}
4. I found this article: {evidence}
5. This suggests
""".strip()

EVAL_BY_COMMON_QUES_PROMPT = """Suppose I will give you a target sentence, target location and a list of common questions. Your task is to verify whether the answer to each of the common questions in the context of the target location contained in the target sentence? Note that there could be multiple correct answers to a common question in the context of the target location. If you think that the target sentence contains the answer for this common question in the context of target location then assign a score of 1 and else assign the score of 0. For each common question, you will calculate the score. For example:

Target sentence: A train derailment occurred on February 3, 2023, at 8:55 p.m. IST, when 38 cars of a Vizianagaram freight train carrying hazardous materials derailed in Andhra Pradesh, India.
Target location: Andhra Pradesh
Common questions: {{"ques_0": "Can you mention a train accident?", "ques_1": "Name an accident which occured due to train derailment?"}}
Scores: {{"ques_0": 1, "ques_1": 1}}

Target sentence: The Indian Institute of Science is a prestigious research university specially in the field of science located in Bangalore, India. Established in 1909, the Indian Institute of Science was the second Indian university based on the European research institution model.
Target location: India
Common questions: {{"ques_0": "Can you give an example of a engineering research focused university?", "ques_1": "Can you provide name of a university which introduced something new?"}}
Scores: {{"ques_0": 0, "ques_1": 0}}

Target sentence: Amitabh Bachchan is an Indian actor and producer. He is widely regarded as one of India's leading actors, having appeared in a wide range of films in the protagonist role.
Target location: India
Common questions: {{"ques_0": "Can you name an actor who is widely regarded as one of the country's leading actors?", "ques_1": "Name an actor who has appeared in a wide range of films in an antagonist role."}}
Scores: {{"ques_0": 1, "ques_1": 0}}

Target sentence: {target_claim}
Target location: {target_location}
Common questions: {common_ques}
For the abpve target sentence, target location and common questions, the score would be:
""".strip()

EVAL_BY_SINGLE_COMMON_QUES_PROMPT = """Suppose I will give you a target sentence, target location and a common question. Your task is to verify whether the answer to the common question in the context of the target location contained in the target sentence? Note that there could be multiple correct answers to a common question in the context of the target location. If you think that the target sentence contains the answer for this common question in the context of target location then assign a score of 1 and else assign the score of 0. Do not include this prompt in your response. Only provide the string which starts with 'Score:' followed by just a single binary score either 1 or 0 and NOTHING ELSE. You must provide only ONE binary score for the question. For example:

Target sentence: A train derailment occurred on February 3, 2023, at 8:55 p.m. IST, when 38 cars of a Vizianagaram freight train carrying hazardous materials derailed in Andhra Pradesh, India.
Target location: Andhra Pradesh
Common question: "Can you mention a train accident?"
Score: 1

Target sentence: The Indian Institute of Science is a prestigious research university specially in the field of science located in Bangalore, India. Established in 1909, the Indian Institute of Science was the second Indian university based on the European research institution model.
Target location: India
Common question: "Can you give an example of a engineering research focused university?"
Score: 0

Target sentence: Amitabh Bachchan is an Indian actor and producer. He is widely regarded as one of India's leading actors, having appeared in a wide range of films in the protagonist role.
Target location: India
Common question: "Can you name an actor who is widely regarded as one of the country's leading actors?"
Score: 1

Target sentence: {target_claim}
Target location: {target_location}
Common question: {common_ques}
For the above target sentence, target location and common question, the score would be:
""".strip()

EVAL_BY_SINGLE_COMMON_QUES_PROMPT_2 = """Suppose I will give you a target sentence, target location and a common question. Your task is to verify whether the answer to the common question in the context of the target location contained in the target sentence? Note that there could be multiple correct answers to a common question in the context of the target location. If you think that the target sentence contains the answer for this common question in the context of target location then assign a score of 1 and else assign the score of 0. Do not include this prompt in your response. Only provide the string which starts with 'Score:' followed by just a single binary score either 1 or 0 and NOTHING ELSE. You must provide only ONE binary score for the question. For example:

Target sentence: IndiGo is a major Indian low-cost airline headquartered in Gurgaon, Haryana, India. IndiGo operates scheduled flights throughout India, Middle East, Asia, Southeast Asia, and Europe.
Target location: India
Common question: "What is the name of a major low cost airline?"
Score: 1

Target sentence: Funtasia Water Park is a water park in Patna. The company currently operates a single location in Patna.
Target location: Patna
Common question: "Can you give an example of an amusement park which is located in the state capital?"
Score: 1

Target sentence: Awarded during the Indian Sports Honors, the Arjuna Award is considered to be the second most prestigious individual prize in sports.
Target location: India
Common question: "What is considered the highest individual prize in a sport?"
Score: 0

Target sentence: {target_claim}
Target location: {target_location}
Common question: {common_ques}
For the above target sentence, target location and common question, the score would be:
""".strip()


SINGLE_EDITOR_PROMPT = """This task involves processing a claim by attributing it based on a set of evidences. The aim is to refine the initial claim into an attributed claim that incorporates insights from all provided evidences.

Instructions:

1. Identify the main entity discussed in the provided claim. Carefully review all associated evidences. Note that the evidences may or may not be relevant to the main entity of the claim. 
2. Determine the relevance of each piece of evidence to the main entity in the claim. Synthesize the factual information from relevant evidences to assess how they support, refute, or modify the initial claim. 
3. Generate an attributed claim that effectively integrates the initial claim with the relevant evidences, ensuring that the main entity of the claim remains unchanged, especially in the context of any irrelevant evidence. 
4. Do not include unnecessary evidence sentences in the modified claim which were not present in the original claim. You are required to check only the factual correctness of the claim without adding extra information to the claim. 

Example:

Claim: Tata Motors is an Indian multinational automobile manufacturing company headquartered in Mumbai, Maharashtra, India. It was established in 1954.
Evidences:
1. Mahindra & Mahindra Limited (M&M) is an Indian multinational automotive manufacturing corporation headquartered in Mumbai. It was established in 1945 as Mahindra & Mohammed and later renamed Mahindra & Mahindra.
2. Tata Motors was founded in 1945, as a locomotive manufacturer. Tata Group entered the commercial vehicle sector in 1954 after forming a joint venture with Daimler-Benz of Germany in which Tata developed a manufacturing facility in Jamshedpur for Daimler lorries.
Attributed Claim: Tata Motors is an Indian multinational automobile manufacturing company headquartered in Mumbai, Maharashtra, India. It was established in 1945.

Claim: Feluda is a detective novel written by renowned Bengali actor Sandip Ray, first published in West Bengal in 1965 by Ananda Publishers. The book has been adapted into a film and several television series.
Evidences:
1. Feluda is an Indian-Bengali detective media franchise created by Indian-Bengali film director and writer Satyajit Ray, featuring the character, Feluda.
2. In 1965, at the age of 44, soon after the release of his landmark film Charulata, Satyajit Ray wrote the first draft of a short story, which featured a young boy, barely into his teens, describing the superlative analytical and detection powers of his older cousin brother."
Attributed Claim: "Feluda is a detective novel written by renowned Bengali author Satyajit Ray, first published in West Bengal in 1965 by Ananda Publishers. The book has been adapted into a film and several television series.

Claim: Leonardo DiCaprio won his first Oscar for Best Actor for his role in the film 'Titanic' in 1996.
Evidences:
1. Leonardo DiCaprio has been nominated for the Best Actor Oscar multiple times, beginning with his role in 'What's Eating Gilbert Grape' in 1993.
2. DiCaprio's performance in 'The Revenant' was universally acclaimed, and he won the Academy Award for Best Actor in 2016, which was his first Oscar win.
3. Leonardo DiCaprio is an active environmentalist who has donated millions to conservation efforts.
Attributed Claim: Leonardo DiCaprio won his first Oscar for Best Actor for his role in 'The Revenant' in 2016, after several nominations for other films including his first for 'What's Eating Gilbert Grape.' 

Claim: Avengers: Endgame was released worldwide in April 2018 and became the highest-grossing film of all time by surpassing 'Titanic'.
Evidences:
1. Avengers: Endgame was released in April 2019. It quickly garnered acclaim for its dramatic conclusion of the Infinity Saga."
2. In July 2019, 'Avengers: Endgame' surpassed 'Avatar' to become the highest-grossing film ever, a record it held until 'Avatar' reclaimed the title after a re-release."
3. The soundtrack for 'Avengers: Endgame' was composed by Alan Silvestri, who also composed music for 'Back to the Future.'"
Attributed Claim: Avengers: Endgame was released worldwide in April 2019 and became the highest-grossing film of all time by surpassing 'Avatar' in July of that year, although 'Avatar' later reclaimed the top spot.

For this claim and evidences, generate the attributed claim as instructed.
Claim: {claim}
Evidences: {evidences}
Attributed Claim: 
""".strip()