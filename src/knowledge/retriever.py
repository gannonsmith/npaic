import re
from src.knowledge.graph_builder import KnowledgeGraph

class KnowledgeGraphRetriever:
    def __init__(self):
        self.kg = KnowledgeGraph()

    @staticmethod
    def _extract_characters_from_context(context, excluded_nodes):
        matches = re.findall(r'<([^/]+?)>', context)
        extracted = {name.strip() for name in matches if name.strip() not in excluded_nodes}
        return list(extracted)

    def get_relevant_facts(self, mission: str, context: str, speaker: str, target: str):
        relevant_facts = []

        # Arthur's perspective on speaker
        for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
            if t == speaker:
                fact = f"{s} {data['type']} {t}. Details: {data.get('details', 'N/A')}"
                relevant_facts.append(fact)
        
        # Speaker-Target relationship (Direct connection, mainly Arthur's involvement)
        if self.kg.graph.has_node(speaker) and self.kg.graph.has_node(target):
            # Check for edge: Speaker -> Target
            for s, t, data in self.kg.graph.edges(speaker, data=True):
                if t == target:
                    fact = f"Direct Fact: {s} {data['type']} {t}. Details: {data.get('details', 'N/A')}"
                    relevant_facts.append(fact)
            
            # Check for edge: Target -> Speaker
            for s, t, data in self.kg.graph.edges(target, data=True):
                if t == speaker:
                    fact = f"Direct Fact: {s} {data['type']} {t}. Details: {data.get('details', 'N/A')}"
                    relevant_facts.append(fact)

        # Mission Context (What is the Mission about, and Arthur's involvement)
        # Search: Mission -> Relationship -> Entity
        for s, t, data in self.kg.graph.edges(mission, data=True):
            fact = f"Mission Fact: {s} {data['type']} {t}. Details: {data.get('details', 'N/A')}"
            relevant_facts.append(fact)

        # Search: Arthur Morgan -> Relationship -> Mission
        for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
            if t == mission:
                fact = f"Arthur's Mission Involvement: {s} {data['type']} in mission {t}. Details: {data.get('details', 'N/A')}"
                relevant_facts.append(fact)
        
        excluded = {speaker, target, "Arthur Morgan", "action"}
        context_characters = self._extract_characters_from_context(context, excluded)

        for char in context_characters:
            if self.kg.graph.has_node(char):
                for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
                    if t == char:
                        fact = f"Context Fact (Arthur's View): {s} {data['type']} {t}. Details: {data.get('details', 'N/A')}"
                        relevant_facts.append(fact)

        return list(set(relevant_facts))

def main():
    example_data = {
        "mission": "Who the Hell is Leviticus Cornwall?",
        "context": "<Arthur Morgan> Yeah. </Arthur Morgan> <Dutch van der Linde> You wanna head down? See how he's getting on? </Dutch van der Linde> <Arthur Morgan> Okay. </Arthur Morgan> <action> [He rides to Bill, who plants explosives on railroad tracks.] </action> <Arthur Morgan> How you getting on? </Arthur Morgan> <Bill Williamson> Yeah... I'm okay. </Bill Williamson> <Arthur Morgan> You sure? </Arthur Morgan> <Bill Williamson> Of course. </Bill Williamson> <Arthur Morgan> Can I help a little? </Arthur Morgan> <Bill Williamson> Alright. Go ahead... and set up the detonator by those rocks over there. </Bill Williamson>",
        "speaker": "Bill Williamson",
        "utterance": "Alright. Go ahead... and set up the detonator by those rocks over there.",
        "response_speaker": "Arthur Morgan",
        "response": "Okay, sure.",
        "gold_response_action": "none"
    }

    # 6. Initialize the Retriever and Get Facts
    retriever = KnowledgeGraphRetriever()
    facts = retriever.get_relevant_facts(
        mission=example_data.get("mission"),
        context=example_data.get("context"),
        speaker=example_data["speaker"],
        target=example_data["response_speaker"]
    )

    # 7. Print the Results
    print("\n--- Facts Retrieved for the Example JSON (Context Used) ---")
    for fact in facts:
        print(f"- {fact}")

if __name__ == "__main__":
    main()