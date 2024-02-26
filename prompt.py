urban_direct_prompt = "Word: {}\nExample: {}\nMeaning:"
urban_instruction_prompt = """
Instruction: Given a word or a phrase, use the provided example to infer the meaning of the word or the phrase. When analyzing the usage examples, interpret the context literally and think through it carefully to infer the meaning of the word (or the phrase). If the meaning cannot be inferred from the example, it is appropriate to say, "The meaning is unclear."
Word: {word}
Example: {example}
Analysis:
Meaning:
"""
urban_icl_prompt = """
Word: "meet ugly"
Example: "When Min Ho bumped into Kitty in the 1st episode, it was the meet ugly of the season, especially because they were enemies to almost lovers."
Meaning: The first encounter between individuals that is unpleasant or unfavorable, often leading to a complex relationship.

Word: "Slugging"
Example: "If your skin is dry after washing, you might add your hyaluronic acid after a rose water spritz, followed by your night creme, and finish with Vaseline as your 'slug' to trap the moisture."
Meaning: A skincare technique where a thick, occlusive layer (like Vaseline) is applied over other skincare products to lock in moisture, resulting in a dewy skin finish.

Word: "Fag"
Example: "See also faggot, twinky, and queer. Never gay. Unless said person is happy (the fag felt gay) which is the proper meaning of gay (gaeity, happy, cheerful)."
Meaning: A derogatory term used to describe a person, often related to their sexual orientation or behavior, and sometimes associated with judgments about their character or actions.

Word: {word}
Example: {example}
Meaning:
"""


urban_cot_prompt = """Instruction: Given a phrase, use the provided example to deduce the meaning of the phrase (or word). It's important to analyze both literal definition and contextual usage of the phrase.

Note: 
1. Explain the phrase in simple words that everyone, even kids, can understand. If needed, use more words to make sure all meanings are clear. Use easy words to explain the phrase clearly.
2. Write like a dictionary but very simple and easy to read.

A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for generating meanings to avoid irreversible consequences.

Word: {word}
Example: {example}
Step by step analysis:
Conclusion: ...
"""

urban_causal_prompt = """Instruction: Given an example with a masked word (or phrase, could be internet slang or memes), analyze the context and infer the meaning of the masked word (or phrase). Think through the example carefully to deduce the possible meaning. If the meaning cannot be inferred from the context, it is appropriate to say, "The meaning is unclear."
Example with Masked Word: {}
Analysis:
Inferred Meaning:
"""



urban_causal_mid_prompt = """Phrase: {word}

Example: {example}

Tasks:
1. Random Entity Replacement:
   - Randomly choose some entities related to "{word}" in the example.
   - Create new entities that fit in the context of "{word}".

2. Sentence Reconstruction:
   - Insert "[MASKED_PHRASE]" in place of "{word}".
   - Replace the chosen entities with the newly created ones.
   - Ensure grammatical and logical coherence.

Output format:
Entity:
New entity:
Reconstructed Sentence:

"""

urban_causal_final_prompt = """Instruction: Given a phrase, use the provided example to deduce the meaning of the phrase (or word). When reviewing usage examples, interpret the context thoroughly to infer the nuanced meaning of the phrase. Break down your reasoning into step-by-step logic to arrive at a comprehensive understanding.

Phrase:{}
Usage example:{}

1. Direct Interpretation: This is an explanation of the phrase based on its usage in a sentence. 
   - Possible Error: Misinterpretation of context or literal meaning. 
   - Interpretation: {}

2. Contextual Inference: This is the meaning inferred from a sentence where the phrase is masked or implied.
   - Possible Error: Incorrect inference due to lack of context or ambiguity.
   - Inferred Meaning: {}

3. Reconstructed Context Inference: This is the meaning inferred from a sentence where related entities are replaced while maintaining the same relationship with the phrase.
   - Possible Error: The new entities might not perfectly mimic the original context, leading to a skewed understanding.
   - Inferred Meaning: {}

Task: Synthesize a comprehensive definition or explanation of the phrase, considering the potential errors and key insights from each step. 

Note: 
1. Explain the phrase in simple words that everyone, even kids, can understand. If needed, use more words to make sure all meanings are clear. Use easy words to explain the phrase clearly.
2. Write like a dictionary but very simple and easy to read.

A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for generating definitions to avoid irreversible consequences.

Output example:
Phrase: Slugging
Definition: Slugging refers to a skincare technique where an emollient or occlusive is applied to trap moisture. It's often used after washing the face to achieve a dewy look. This expression describes a specific beauty routine.

Phrase: Coviding
Definition: Coviding refers to the practice of adopting measures to prevent COVID-19 infection. It's often used in the context of social and personal health safety. This expression signifies cautious behavior during the pandemic.

Phrase: The Winter Arc
Definition: The Winter Arc refers to a period of facing winter's mental and physical challenges. It's often used to describe a time of perseverance and productivity. This expression implies resilience and determination.

Definition:
"""


urban_causal_legacy_propose_prompt = """Phrase: {word}

Example: {example}

Tasks:
1. Entity Candidate Generation:
   - For each entity in the example, including the Phrase, generate three alternative entities.
   - Include the original entity as a fourth candidate.

2. Sentence Reconstruction:
   - Replace "{word}" with "[MASKED_PHRASE]".
   - Replace each entity in the sentence with "[MASKED_ENTITY_1]", "[MASKED_ENTITY_2]", etc., in sequence.

Output:

1. Entity Replacement List:
   {{
     "[MASKED_PHRASE]": Candidate 1, Candidate 2, Candidate 3, Original Phrase;
     "[MASKED_ENTITY_1]": Candidate 1, Candidate 2, Candidate 3, Original Entity 1;
     ...
   }}

2. Reconstructed Sentence: (Sentence with "[MASKED_PHRASE]" and "[MASKED_ENTITY_X]")
"""

urban_causal_legacy_cot_prompt = """Instruction: Given a phrase and a reconstructed example sentence, use the reconstructed example to infer the meaning of the phrase. Additionally, consider the "Entity Candidates List". This list is crucial as it provides all possible entities or interpretations for each specific position in the original example sentence. By exploring these alternatives, you can gain a deeper understanding of the different contexts and nuances in which the phrase might be used. Methodically break down your reasoning into steps to arrive at a clear and well-explained conclusion.

Note:
1. Explain the phrase in simple words that everyone, including kids, can understand. Use more words if necessary to ensure clarity, and use easy words for explanations.
2. Write like a dictionary but in a very simple and easy-to-read manner.

Template for defining meanings, considering the variations of the phrase's use:
"[Word] refers to [basic description based on the reconstructed example]. In different contexts, it could mean [interpretations based on the entity candidates list]. It is often used [context or situation of usage based on the reconstructed example and entity candidates]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."

Warning: Please strictly follow this template for generating meanings to avoid irreversible consequences.

Phrase: {phrase}
Original Example: {example}
Reconstructed Example: {reconstructed_example}
Entity Candidates List: {entity_candidates_json}
   - This list contains alternative entities or interpretations for each entity in the original example. It offers insights into various possible meanings and uses of the phrase in different contexts.

Examples:
Conclusion: "[MASKED_PHRASE] refers to a skincare technique where an emollient or occlusive is applied to trap moisture. It's often used after washing the face to achieve a dewy look. This expression describes a specific beauty routine."

Conclusion: "[MASKED_PHRASE] refers to the practice of adopting measures to prevent COVID-19 infection. It's often used in the context of social and personal health safety. This expression signifies cautious behavior during the pandemic."

Conclusion: "[MASKED_PHRASE] refers to a period of facing winter's mental and physical challenges. It's often used to describe a time of perseverance and productivity. This expression implies resilience and determination."

Output:
A: Step 1: Analyze the reconstructed example and entity candidates.
   Step 2: Infer the various meanings and contexts for the phrase.
   Conclusion: ...
"""

model_knowledge_check = """
As part of an academic research project, I am evaluating phrases from Urban Dictionary to determine their uniqueness in relation to your training data as a large language model.

Phrase to Evaluate: {word}
Original Explanation: {meaning}
Original Usage Example: {example}

Honesty Reminder:
- You are reminded to be honest in your response. If you are not familiar with the phrase "{word}" and its specific meaning as provided, please respond with "I do not know" rather than guessing or making assumptions.

Evaluation Criteria:
- Indicate whether you are familiar with the phrase "{word}" and its specific meaning as outlined in the provided explanation.
- If you are already familiar with this phrase and its meaning, then the recommendation will be to discard it from the research dataset.
- If this phrase and its meaning are not known to you, or if they were not part of your training data, then the recommendation will be to retain it in the research dataset.

Ouptut format:

Decision: [Retain/Discard]
Explanation: [Brief explanation here]

Note: Replace [Retain/Discard] and [Brief explanation here] with your specific decision and explanation respectively, based on your familiarity with the phrase.
"""


clean_data = """
Given the phrase (or word) "{word}" from Urban Dictionary, along with its usage example "{example}" and meaning "{meaning}", evaluate the content for its suitability in our dataset by following these steps:

1. Examine for NSFW Content:
   - Check for Explicit Language in Example: Does the usage example contain sexually explicit language or vivid adult content? (Yes/No)
   - Assess for Violence in Example: Are there descriptions or glorifications of violence in the usage example? (Yes/No)

2. Assess Aggressiveness in Meaning:
   - Check for Extreme Hate Speech in Meaning: Does the definition include severe forms of hate speech or incitement to violence against specific groups? (Yes/No)

3. Final Evaluation:
   - Retention Criteria: The content should be retained unless any of the answers to the above questions is "Yes".
   - Discard Criteria: The content should be discarded if any answer to the above questions is "Yes".
   - Final Decision: Should the content be retained or discarded? (Retain/Discard)

4. Decision Justification:
   - Provide a brief explanation for your decision. This explanation will help understand the context and rationale behind your decision.

Ouptut format:

Decision: [Retain/Discard]
Explanation: [Brief explanation here]

Note: Replace [Retain/Discard] and [Brief explanation here] with your specific decision and explanation respectively.
"""

transform_dict_data = """Change internet slang or memes into simple dictionary entries. Make sure they're easy to understand:

Input: 
- Phrase: {word}
- Original Explanation: {meaning}
- Original Example: {example}

Process:
1. Explain the phrase in simple words that everyone, even kids, can understand. If needed, use more words to make sure all meanings are clear.
2. Provide a concise example that demonstrates the usage of the phrase. The example should subtly convey the true meaning of the phrase through its semantic context, emotions, and cultural background, allowing the meaning to be inferred without explicitly explaining it (DO NOT explain it directly in usage example!!!).
3. Keep the real meaning of the phrase as people know it online.
4. Write like a dictionary but very simple and easy to read.

Explanation Transformation Guidelines:
A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for conversion to avoid irreversible consequences.

Explanation Transformation Example:

Phrase: Slugging
Original Explanation: "Involves coating the skin in some kind of emollient or occlusive as a way of trapping moisture, for a dewy, slimy look."
Transformed Explanation: "Slugging refers to a skincare technique where an emollient or occlusive is applied to trap moisture. It's often used after washing the face to achieve a dewy look. This expression describes a specific beauty routine."

Phrase: Coviding
Original Explanation: "Taking precautions to prevent the infection and spread of COVID-19."
Transformed Explanation: "Coviding refers to the practice of adopting measures to prevent COVID-19 infection. It's often used in the context of social and personal health safety. This expression signifies cautious behavior during the pandemic."

Phrase: The Winter Arc
Original Explanation: "A time to face challenges of winter and get things done."
Transformed Explanation: "The Winter Arc refers to a period of facing winter's mental and physical challenges. It's often used to describe a time of perseverance and productivity. This expression implies resilience and determination."      

Example Transformation Guidelines:
1. Clear Example: Create an example that uses the phrase in a way that's easy to understand.
2. True to Online Use: Make sure the example match how the phrase is used online, and the example should conform to the cultural background and true meaning (both original and transformed explaination) of this phrase.
3. Ensure that the entire phrase appears in transformed example completely.

Output:
- Analysis: [Talk about how you made the explanation and example clear, true to online culture, and easy to understand]
- Transformed Explanation: [Simple and clear explanation of the phrase]
- Transformed Usage Example: [Simple example that shows how the phrase is used, helping to understand its meaning]

Note: Fill in the placeholders with real internet slang or meme information. Use language that is easy, common, and straightforward in both the explanation and example.]"""

generate_new_meaning = """Input:

Phrase: {word}
Explanation: {transformed_meaning}

Task: 

I've provided you with an explanation of the phrase in the dictionary. Rewrite the explanation to preserve its original intent and meaning. Your goal is to maintain a similar sentence length and structure but use different wording. Focus on substituting words and phrases with their synonyms while ensuring the meaning remains clear and consistent with the original explanation. Avoid altering the fundamental concepts and essential terms.
A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for conversion (NEVER modify words that already appear in the template!) to avoid irreversible consequences.

Phrase: Fusk
Explanation: Fusk refers to a word used to show that one is really annoyed. It is often used among friends or group members who understand and support each other. This expression shows strong feelings without using bad language.\"",
            
New Transformed Explanation 1: Fusk refers to a term signaling that someone is extremely irritated. It is often used among companions or team members who share mutual understanding and camaraderie. This phrase conveys intense emotion without resorting to offensive language.
New Transformed Explanation 2: Fusk refers to an expression indicating significant annoyance. It is often used within circles of acquaintances or allies familiar with each other's sentiments. This expression communicates deep emotions while avoiding vulgar terms.
New Transformed Explanation 3: Fusk refers to a slang that denotes a high level of vexation. It is often used among peers or in-group contexts where there's a strong sense of fellowship. This term expresses powerful feelings without the use of profanity.
New Transformed Explanation 4: Fusk refers to a colloquialism used to express considerable exasperation. It is often used by a cluster of friends or group affiliates who are supportive and empathetic towards one another. This locution articulates vehement emotions without the employment of crude words.
New Transformed Explanation 5: Fusk refers to a word used to show that one is really annoyed. It is often used among friends or group members who understand and support each other. This expression shows strong feelings without using bad language.

Output:

New Transformed Explanation 1: [Your newly crafted explanation]
New Transformed Explanation 2: [Your newly crafted explanation]
New Transformed Explanation 3: [Your newly crafted explanation]
New Transformed Explanation 4: [Your newly crafted explanation]


Note: Replace the placeholders in brackets [ ] with your content. New Transformed Explanation should start with {word}.
"""