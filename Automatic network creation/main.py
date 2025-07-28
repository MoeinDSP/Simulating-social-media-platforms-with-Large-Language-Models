import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
import openai  # You'll need to install this: pip install openai
import time
from huggingface_hub import InferenceClient
import os
import csv
import pandas as pd

# Root directory for all outputs
model_name = "mistral"
ROOT_DIR = os.path.join("conversations1", model_name)

# Create necessary directories if they don't exist
os.makedirs(os.path.join(ROOT_DIR, "raws"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "jsons"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "csv"), exist_ok=True)

@dataclass
class Persona:
    name: str
    gender: str
    age: int
    economic_status: str
    occupation: str

@dataclass
class ConversationConfig:
    topic: str
    max_messages_per_person: int
    max_message_length: int
    tone: str  # e.g., "casual", "formal", "debate", "friendly"
    num_participants: int  # Number of personas to include in the conversation
    personas: List[Persona]

class ConversationSimulator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.hf_client = InferenceClient(
            provider="together",
            api_key="",  # Replace with your actual API key
        )
        self.hf_client2 = InferenceClient(
            provider="together",
            api_key="",  # Replace with your actual API key
        )
        
        # Create CSV directory
        os.makedirs(os.path.join(ROOT_DIR, "csv"), exist_ok=True)

    def save_raw_conversation(self, response: str, config: ConversationConfig):
        print(response)
        conv_name = f"{config.topic}"
        with open(os.path.join(ROOT_DIR, "raws", f"{conv_name}.txt"), "w", encoding='utf-8') as file:
            file.write(response)
        
    def invoke_llm(self, prompt):
        response  = self.client.chat.completions.create(
            model="gpt-4",  # or another appropriate model
            messages=[
                {"role": "system", "content": "You are a social media conversation simulator. Generate realistic conversations between the given personas. IMPRTANT NOTE: Incorporate the personas characteristics so that the messages of each reflect their personas"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        response = response.choices[0].message.content
        return response.strip()
    
    def invoke_llm_hf2(self, prompt, model_id="mistralai/Mistral-7B-Instruct-v0.3"):
        # Convert the prompt into a chat format
            # For zero-shot, simpler format
        completion = self.hf_client2.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for more deterministic responses
            # max_tokens=50
        )
        
        # Extract the response text
        response = completion.choices[0].message.content.strip()
        return response
    
    def invoke_llm_hf(self, prompt):
        response = self.hf_client.text_generation(
            prompt=prompt,
            temperature=0.1,  # Low temperature for more deterministic responses
            # repetition_penalty=1.05
        )
        return response.strip()
            
    def generate_conversation(self, config: ConversationConfig) -> List[Dict]:
        # Select the specified number of participants
        selected_personas = config.personas
        
        # Create a prompt that describes the personas and conversation parameters
        prompt = self._create_prompt(selected_personas, config)
        
        # Call the LLM to generate the conversation
        response = self.invoke_llm_hf2(prompt)
        
        self.save_raw_conversation(response, config)
        
        # Parse and structure the conversation
        messages = self._parse_conversation(response)
        
        # Set recipients for each message
        participant_names = [p.name for p in selected_personas]
        for msg in messages:
            # Recipients are all participants except the sender
            msg['recipients'] = [name for name in participant_names if name != msg['sender']]
            
        return messages
    
    def _create_prompt(self, personas: List[Persona], config: ConversationConfig) -> str:
        persona_descriptions = []
        for persona in personas:
            description = f"""
            Name: {persona.name}
            Gender: {persona.gender}
            Age: {persona.age}
            Economic Status: {persona.economic_status}
            Occupation: {persona.occupation}
            """
            persona_descriptions.append(description)
        
        prompt = f"""
        Generate a social media conversation between these personas:
        
        {''.join(persona_descriptions)}
        
        Conversation parameters:
        - Topic: {config.topic}
        - Maximum messages PER PERSON: {config.max_messages_per_person}
        - Maximum message length: {config.max_message_length} characters
        - When Creating the messages, incorporate the personas characteristics so that the messages of each reflect their personas.
        
        Return the conversation in the following JSON format:
        {{
            "conversation": [
                {{
                    "persona": "name",
                    "message": "message content",
                    "tone": "message tone"
                }}
            ]
        }}
        """
        return prompt
    
    def _parse_conversation(self, llm_response: str) -> List[Dict]:
        try:
            # Extract JSON from the response
            conversation_json = json.loads(llm_response)
            
            # Transform to new format
            messages = []
            for msg in conversation_json.get('conversation', []):
                # Use the persona name directly
                if msg['persona'] in ["Kayla", "Morgan", "Frank", "Karen", "Leo"]:  # Only include messages from known personas
                    messages.append({
                        "sender": msg['persona'],
                        "recipients": "all",  # Will be set by scenario later
                        "message": msg['message']
                    })
            return messages
        except json.JSONDecodeError:
            # If the response isn't valid JSON, try to extract it
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                return self._parse_conversation(json_str)  # Recursively parse the extracted JSON
            return []

    def _get_agent_id(self, name: str) -> Optional[int]:
        # Map persona names to their IDs
        name_to_id = {
            "Kayla": 0,
            "Morgan": 1,
            "Frank": 2,
            "Karen": 3,
            "Leo": 4
        }
        return name_to_id.get(name)

    def convert_json_to_csv(self, messages: List[Dict], topic: str):
        """
        Convert conversation messages to CSV format and save it.
        
        Args:
            messages (List[Dict]): The list of conversation messages
            topic (str): The topic of the conversation
        """
        try:
            # Create DataFrame directly from messages list
            df = pd.DataFrame(messages)
            
            # Save to CSV with UTF-8 encoding
            csv_path = os.path.join(ROOT_DIR, "csv", f"{topic}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV saved successfully at: {csv_path}")
            
        except Exception as e:
            print(f"Error converting JSON to CSV: {str(e)}")


def create_scenario(personas, topic, max_messages_per_person=3, max_length=350):
    max_messages_per_person = random.randint(10, 20)
    return ConversationConfig(
        topic=topic,
        max_messages_per_person=max_messages_per_person,
        max_message_length=max_length,
        num_participants=len(personas),
        tone='None',
        personas=personas
    )


def main():
    # Example usage with diverse personas
    personas = [
        Persona(name="Kayla", gender="Female", age=16, economic_status="Working Class", occupation="TikTok Influencer"),
        Persona(name="Morgan", gender="Nonbinary", age=30, economic_status="Upper-Middle", occupation="Corporate Lawyer"),
        Persona(name="Frank", gender="Male", age=55, economic_status="Poor", occupation="Uber Driver"),
        Persona(name="Karen", gender="Female", age=45, economic_status="Middle Class", occupation="Politician (Controversial)"),
        Persona(name="Leo", gender="Male", age=24, economic_status="Lower Class", occupation="Activist (Environmental)"),
    ]
    
    A, B, C, D, E = personas[0], personas[1], personas[2], personas[3], personas[4]
    
    # Initialize the simulator
    API_KEY = ''
    simulator = ConversationSimulator(api_key=API_KEY)
    
    # Creative scenarios
    scenarios = [
        # Social Media and Digital Culture
        create_scenario([A, B], "Cancel Culture and Social Media", max_messages_per_person=17),
        create_scenario([A, B], "Gender Pay Gap", max_messages_per_person=12),
        create_scenario([A, B, C], "Social Media Addiction", max_messages_per_person=15),
        create_scenario([A, B, C, D], "Gun Control Laws", max_messages_per_person=12),
        create_scenario([A, B, C, D], "Systemic Racism in Society", max_messages_per_person=10),
        create_scenario([B, C], "Gig Economy Workers' Rights", max_messages_per_person=12),
        create_scenario([B, C, D], "Corporate Social Responsibility", max_messages_per_person=20),
        create_scenario([B, C, D], "Green Energy Transition", max_messages_per_person=15),
        create_scenario([B, D], "Affirmative Action in Education", max_messages_per_person=19),
        create_scenario([D, E], "Environmental Protests vs. Economic Impact", max_messages_per_person=17),
    ]
    
    # Generate conversations for each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\nGenerating conversation for scenario {i+1}: {scenario.topic}")
        messages = simulator.generate_conversation(scenario)
        
        # Get participant IDs for filename
        participant_ids = [simulator._get_agent_id(p.name) for p in scenario.personas]
        filename = f"{participant_ids}-{scenario.topic.replace(' ', '_')}-{scenario.max_messages_per_person}.json"
        
        # Save JSON with UTF-8 encoding
        with open(os.path.join(ROOT_DIR, "jsons", filename), "w", encoding='utf-8') as file:
            json.dump(messages, file, indent=2, ensure_ascii=False)
            
        # Convert and save CSV
        simulator.convert_json_to_csv(messages, scenario.topic)
        
        # Optional: Add a small delay between API calls to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    main()
