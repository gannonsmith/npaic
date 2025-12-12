import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_graph()

    def add_entity(self, name, entity_type, description=None):
        if name not in self.graph:
            self.graph.add_node(name, type=entity_type, description=description)

    def add_relationship(self, source, target, relationship_type, details=None):
        if source in self.graph and target in self.graph:
            existing_edges = self.graph.get_edge_data(source, target, default={})
            self.graph.add_edge(source, target, type=relationship_type, details=details)
        else:
            print(f"Error: One of both entities '{source}' or '{target}' not found.")


    def visualize_graph(self):
        color_map = {
            'Character': 'skyblue',
            'Location': 'lightgreen',
            'Mission': 'lightcoral'
        }

        node_colors = [color_map.get(self.graph.nodes[n].get('type'), 'gray')
                       for n in self.graph.nodes()]
        
        pos = nx.spring_layout(self.graph, k=0.15, iterations=20)

        plt.figure(figsize=(12, 12))

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=self.graph.edges(),
            arrowstyle="->",
            arrowsize=20,
            edge_color='gray'
        )

        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=10,
            font_weight='bold'
        )

        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_color='darkred'
        )

        plt.title("Arthur Morgan's RDR2 Knowledge Graph", fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("results/figures/kg.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _populate_graph(self):
        ## Character graph
        entities = [
            ("Arthur Morgan", "Character", "Protagonist, loyal enforcer, whose loyalty is tested by Dutch."),
            ("John Marston", "Character", "A former prodigal son. Old friend, but foolish and unreliable until the very end. Arthur feels a duty to protect his family."),
            ("Dutch van der Linde", "Character", "The leader and Arthur's surrogate father. His erratic behavior and paranoia become Arthur's greatest worry and ultimate disillusionment."),
            ("Sadie Adler", "Character", "A survivor. Became fiercely loyal and capable after Arthur took her in. A reliable, aggressive partner whom Arthur respects deeply."),
            ("Charles Smith", "Character", "A quiet, honorable man. Loyal, capable, and always focused on what is right. Arthur trusts him implicitly, valuing his moral compass."),
            ("Hosea Matthews", "Character", "Arthur's mentor and the true voice of reason. The brains behind the operation. Arthur relies on his wisdom and trusts his judgment above all others."),
            ("Micah Bell", "Character", "A rat and a snake. Trouble personified. Arthur distrusts him immediately and sees him as the physical manifestation of the gang's decline."),
            ("Bill Williamson", "Character", "A simple brute with a short fuse. Loyal to Dutch, but easily led and incompetent. Arthur sees him as more of a nuisance than a true partner."),
            ("Abigail Marston", "Character", "Sharp and resilient. The mother of Jack and John's partner. Arthur respects her strength and eventually takes on the role of her protector."),
            ("Uncle", "Character", "A lazy, rambling drunk. A charming thief and a pain. Despite his uselessness, he's considered family and a fixture of the camp."),
            ("Jack Marston", "Character", "The gang's child, loved and protected by all. Arthur treats him like a younger brother or a son, and his safety becomes Arthur's primary motivation."),
            ("Lenny Summers", "Character", "A smart, young man with potential who found a home in the gang. Arthur saw him as a friend and a good kid whose death was a major emotional blow."),
            ("Javier Escuella", "Character", "Once a poetic idealist, now silently following Dutch into madness. Arthur is disappointed by his blind loyalty and detachment from reality."),
            ("Josiah Trelawny", "Character", "The flamboyant conman and magician. An outsider who brings them work. Arthur views him as a mysterious but useful associate."),
            ("Sean MacGuire", "Character", "The young, mouthy Irishman. A thief and a nuisance, but part of the family. Arthur considered him a friend whose capture and eventual death were a tragic setback."),
        ]
        for name, type, desc in entities:
            self.add_entity(name, type, desc)

        relationships = [
            ("Arthur Morgan", "IS_LOYAL_TO", "Dutch van der Linde", "Lifelong devotion, though conflicted."),
            ("Arthur Morgan", "DOUBTS", "Dutch van der Linde", "Doubts grow after the failures and his erratic behavior."),
            ("Arthur Morgan", "DESPISES", "Micah Bell", "Believes Micah is corrupting Dutch and directly harming the gang."),
            ("Arthur Morgan", "FEELS_BROTHERHOOD_WITH", "Hosea Matthews", "Shared history, intellectual respect, and mutual understanding."),
            ("Arthur Morgan", "TRUSTS", "Hosea Matthews", "Relies on him as the moral and strategic anchor of the gang."),
            ("Arthur Morgan", "FEELS_BROTHERHOOD_WITH", "John Marston", "Despite their friction, they are family."),
            ("Arthur Morgan", "TRUSTS", "Charles Smith", "Values his honor, quiet strength, and reliable counsel."),
            ("Arthur Morgan", "PROTECTS", "John Marston", "Pushes John to leave and save his family after Chapter 6."),
            ("Arthur Morgan", "PROTECTS", "Jack Marston", "His primary motivation for sacrificing his own interests is Jack's future."),
            ("Arthur Morgan", "PROTECTS", "Abigail Marston", "Ensures she and Jack escape to safety."),
            ("Arthur Morgan", "RESPECTS", "Sadie Adler", "Acknowledges her transition from victim to capable fighter."),
            ("Arthur Morgan", "DISAPPOINTED_BY", "Javier Escuella", "Disappointed that Javier chooses to follow Dutch blindly despite the clear truth."),
            ("Arthur Morgan", "IS_WARY_OF", "Josiah Trelawny", "Sees him as an untrustworthy, but useful, associate."),
            ("Arthur Morgan", "IS_WARY_OF", "Bill Williamson", "Finds him incompetent but dangerous when following orders."),
            ("Arthur Morgan", "MOURNS", "Lenny Summers", "Considers his death a tragic loss of a young, promising man."),
            ("Arthur Morgan", "MOURNS", "Sean MacGuire", "Felt personal grief and rage over his execution."),
        ]
        for source, rel_type, target, detail in relationships:
            self.add_relationship(source, target, rel_type, detail)

        ## Locations
        entities = [
            ("Colter", "Location", 
            "The first, miserable hideout. A place of cold, desperation, and early failure."),
            ("Horseshoe Overlook", "Location", 
            "The first stable camp. Arthur felt an early sense of hope and relative safety here."),
            ("Clemens Point", "Location", 
            "The comfortable, idyllic camp near Rhodes. A place of short-lived calm and security."),
            ("Saint Denis", "Location", 
            "The giant, corrupt city. Overwhelming and ultimately the site of the gang's grand failure."),
            ("Shady Belle", "Location", 
            "The temporary hideout near Saint Denis. Too civilized and too close to trouble."),
            ("Beaver Hollow", "Location", 
            "The final, dark, desperate cave camp. A den of madness and decay."),
            ("Doctor's Office", "Location", 
            "The site where Arthur received his tuberculosis diagnosis."),
            ("Mount Hagen", "Location", 
            "The snowy peak where Arthur had his final confrontation and died."),
            ("Blackwater", "Location", 
            "The place where the ferry heist failed, the ghost that haunts the entire story."),
            ("Valentine", "Location", 
            "A key frontier town used for supplies, drinking, and where early conflicts flared up.")
        ]
        for name, type, desc in entities:
            self.add_entity(name, type, desc)

        relationships = [
            ("Arthur Morgan", "CAMPED_AT", "Colter", 
            "Lived temporarily during the blizzard escape."),
            ("Arthur Morgan", "CAMPED_AT", "Horseshoe Overlook", 
            "The first stable base of operations."),
            ("Arthur Morgan", "CAMPED_AT", "Clemens Point", 
            "Enjoyed a brief period of peace and plenty."),
            ("Arthur Morgan", "CAMPED_AT", "Shady Belle", 
            "Too close to the city and the high life that Dutch desired."),
            ("Arthur Morgan", "CAMPED_AT", "Beaver Hollow", 
            "Felt utter despair and observed the gang's final collapse."),
            ("Arthur Morgan", "VISITED_FOR", "Valentine", 
            "Used for early missions, saloons, and commerce."),
            ("Arthur Morgan", "RECEIVED_DIAGNOSIS_AT", "Doctor's Office", 
            "The moment his impending death was confirmed, leading to his moral shift."),
            ("Arthur Morgan", "WITNESSED_FAILURE_AT", "Saint Denis", 
            "The bank robbery and subsequent shootout shattered the gang and led to key deaths."),
            ("Arthur Morgan", "FOUGHT_LAST_STAND_AT", "Mount Hagen", 
            "Location of the final confrontation with Micah and Dutch."),
            ("Arthur Morgan", "FELT_DESPAIR_AT", "Blackwater", 
            "The financial debt and legal heat from this location defined the entire story.")
        ]
        for source, rel_type, target, detail in relationships:
            self.add_relationship(source, target, rel_type, detail)

        ## Mission knowledge
        entities = [
            ("Outlaws from the West", "Mission", 
            "The initial escape mission. Defines the gang's current status: on the run."),
            ("Who the Hell is Leviticus Cornwall?", "Mission", 
            "The first major train robbery. The gang learns they have a powerful enemy."),
            ("A Fisher of Men", "Mission", 
            "Arthur and Dutch meet Colm O'Driscoll, leading to the beginning of the great feud."),
            ("The Sheep and the Goats", "Mission", 
            "A robbery that introduces the Grays and Braithwaites conflict, marking the start of trouble in Rhodes."),
            ("Blessed Are the Meek?", "Mission", 
            "Arthur rescues Micah from jail. An early moment of distrust and poor judgment."),
            ("The Battle of Shady Belle", "Mission", 
            "The mission to take the Shady Belle mansion, establishing a new, more dangerous base."),
            ("The Fine Joys of Civilization", "Mission", 
            "The first mission in Saint Denis. A taste of the high life that distracts Dutch."),
            ("Banking, the Old American Art", "Mission", 
            "The Saint Denis bank robbery. The ultimate failure that resulted in the loss of Hosea and Lenny and forced the gang's exile."),
            ("Sodom? Back to Gomorrah", "Mission", 
            "The train robbery with the young John Marston. Arthur felt they were close and trusted him."),
            ("A Fork in the Road", "Mission", 
            "Arthur discovers Micah's treachery, solidifying his final decision to break with Dutch."),
            ("My Last Boy", "Mission", 
            "Arthur's final act of trying to help Rains Fall. A selfless act that cements his redemption path."),
            ("Red Dead Redemption", "Mission", 
            "Arthur's final mission. Choosing between loyalty (Dutch) or family (John).")
        ]
        for name, type, desc in entities:
            self.add_entity(name, type, desc)

        relationships = [
            ("Outlaws from the West", "ENDED_AT", "Horseshoe Overlook", "Led the gang to their first stable camp."),
            ("The Sheep and the Goats", "TOOK_PLACE_NEAR", "Valentine", "Involved local conflict near the town."),
            ("Blessed Are the Meek?", "TOOK_PLACE_IN", "Valentine", "Arthur broke Micah out of the Valentine jail."),
            ("The Battle of Shady Belle", "ESTABLISHED_CAMP_AT", "Shady Belle", "Resulted in the move to the new camp."),
            ("Banking, the Old American Art", "FOCUSED_ON", "Saint Denis", "The bank robbery was the grand plan in the city."),
            ("Red Dead Redemption", "TOOK_PLACE_AT", "Mount Hagen", "Arthur's final physical location."),
            ("Arthur Morgan", "PARTICIPATED_IN", "Outlaws from the West", "The beginning of the gang's flight."),
            ("Arthur Morgan", "REALIZED_ENEMY_WAS", "Who the Hell is Leviticus Cornwall?", "Realized the power of the enemy chasing them."),
            ("Arthur Morgan", "PARTNERED_WITH", "John Marston", "Sodom? Back to Gomorrah"),
            ("Arthur Morgan", "LOST_FRIEND_DURING", "Banking, the Old American Art", "Hosea and Lenny were killed during this mission."),
            ("Arthur Morgan", "FELT_DISGUST_AFTER", "Blessed Are the Meek?", "Disgusted he had to save Micah."),
            ("Arthur Morgan", "FEELS_REDEMPTION_IN", "My Last Boy", "This was a selfless act to help the Wapiti."),
            ("Arthur Morgan", "DISCOVERED_BETRAYAL_IN", "A Fork in the Road", "The moment he knew Micah was the rat."),
            ("Arthur Morgan", "MADE_FINAL_CHOICE_IN", "Red Dead Redemption", "Chose John's family over Dutch."),
            ("Hosea Matthews", "DIED_DURING", "Banking, the Old American Art", "His death was a major catalyst for Dutch's breakdown."),
            ("Lenny Summers", "DIED_DURING", "Banking, the Old American Art", "His death was a major catalyst for Arthur's grief."),
            ("Micah Bell", "WAS_SAVED_DURING", "Blessed Are the Meek?", "Micah's freedom brought more chaos to the gang."),
        ]
        for source, rel_type, target, detail in relationships:
            self.add_relationship(source, target, rel_type, detail)

if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.visualize_graph()