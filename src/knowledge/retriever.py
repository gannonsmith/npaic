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
    
    @staticmethod
    def _format_fact(source, target, data, prefix=""):
        fact = f"{source} {data['type']} {target}. Details: {data.get('details', 'N/A')}"
        if prefix:
            return f"{prefix}: {fact}"
        return fact

    def get_relevant_facts(self, mission: str, context: str, speaker: str, target: str):
        """
        Retrieves facts that connect the mission and the key characters 
        (speaker, target) from Arthur's perspective, using the context for 
        additional character mentions.
        """
        relevant_facts = []

        # Arthur's perspective on the current SPEAKER (e.g., Bill Williamson)
        # Search: Arthur Morgan -> Relationship -> Speaker
        for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
            if t == speaker:
                relevant_facts.append(self._format_fact(s, t, data))

        # Speaker-Target relationship (Direct connection)
        if self.kg.graph.has_node(speaker) and self.kg.graph.has_node(target):
            # Check for edge: Speaker -> Target
            for s, t, data in self.kg.graph.edges(speaker, data=True):
                if t == target:
                    relevant_facts.append(self._format_fact(s, t, data))
            
            # Check for edge: Target -> Speaker
            for s, t, data in self.kg.graph.edges(target, data=True):
                if t == speaker:
                    relevant_facts.append(self._format_fact(s, t, data))

        # Mission Context (What is the Mission about, and Arthur's involvement)
        # Search: Mission -> Relationship -> Entity
        for s, t, data in self.kg.graph.edges(mission, data=True):
            relevant_facts.append(self._format_fact(s, t, data, prefix="Mission Fact"))

        # Search: Arthur Morgan -> Relationship -> Mission
        for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
            if t == mission:
                relevant_facts.append(self._format_fact(s, t, data, prefix="Arthur's Mission Involvement"))

        # Contextually Relevant Character Facts
        excluded = {speaker, target, "Arthur Morgan", "action"}
        context_characters = self._extract_characters_from_context(context, excluded)
        
        for char in context_characters:
            if self.kg.graph.has_node(char):
                # Search: Arthur Morgan -> Relationship -> Context Character
                for s, t, data in self.kg.graph.edges("Arthur Morgan", data=True):
                    if t == char:
                        relevant_facts.append(self._format_fact(s, t, data, prefix="Context Fact (Arthur's View)"))

        # Remove duplicates using the raw fact string as the key
        return list(dict.fromkeys(relevant_facts))

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