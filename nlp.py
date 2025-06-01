from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Define command templates and their corresponding actions
COMMAND_TEMPLATES = {
    "move left": ["move left", "go left", "move to the left"],
    "move right": ["move right", "go right", "move to the right"],
    "move forward": ["move forward", "go forward", "move ahead"],
    "move back": ["move back", "go back", "move backward"],
    "move up": ["move up", "go up", "move upward"],
    "move down": ["move down", "go down", "move downward"],
    "gripper open": ["open gripper", "open hand", "release grip"],
    "gripper close": ["close gripper", "close hand", "grip"],
    "lift green": ["lift green cube", "pick up green", "grab green"],
    "lift red": ["lift red cube", "pick up red", "grab red"],
    "stack": ["stack cubes", "stack them", "put one on top"],
    "reset": ["reset", "restart", "start over"],
    "break": ["break", "exit", "quit", "stop"]
}

def get_command_embedding(text):
    """Get embedding for a single text."""
    return model.encode(text)

def get_template_embeddings():
    """Get embeddings for all command templates."""
    all_templates = []
    template_to_command = {}
    
    for command, templates in COMMAND_TEMPLATES.items():
        for template in templates:
            all_templates.append(template)
            template_to_command[template] = command
    
    embeddings = model.encode(all_templates)
    return embeddings, all_templates, template_to_command

def process_command(user_input):
    """
    Process user input and return the corresponding command.
    Returns None if no matching command is found.
    """
    user_input = user_input.lower()
    
    # Get embeddings for all templates
    template_embeddings, templates, template_to_command = get_template_embeddings()
    
    # Get embedding for user input
    user_embedding = get_command_embedding(user_input)
    
    # Calculate cosine similarity
    similarities = np.dot(template_embeddings, user_embedding) / (
        np.linalg.norm(template_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )
    
    # Get the most similar template
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[most_similar_idx]
    
    # If similarity is above threshold, return the corresponding command
    if similarity_score > 0.7:  # Adjust threshold as needed
        return template_to_command[templates[most_similar_idx]]
    
    return None

